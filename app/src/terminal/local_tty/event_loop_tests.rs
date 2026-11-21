use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::time::Instant;

use mio::unix::SourceFd;
use parking_lot::Mutex;

use super::*;
use crate::terminal::local_tty::mio_channel;
use crate::terminal::local_tty::{ChildEvent, EventedPty, EventedReadWrite};
use crate::terminal::SizeInfo;

/// How long a test waits for the event loop before calling it stuck.
const TIMEOUT: Duration = Duration::from_secs(10);

fn test_terminal() -> Arc<FairMutex<TerminalModel>> {
    Arc::new(FairMutex::new(TerminalModel::mock(None, None)))
}

/// Output that never runs dry, as from a program like `yes` that writes faster than the
/// terminal parses: every read fills the buffer and none returns `WouldBlock`.
struct EndlessOutput;

impl Read for EndlessOutput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        for (byte, output) in buf.iter_mut().zip(b"y\r".iter().cycle()) {
            *byte = *output;
        }
        Ok(buf.len())
    }
}

/// Records what the event loop writes to the PTY.
#[derive(Clone, Default)]
struct RecordedInput(Arc<Mutex<Vec<u8>>>);

impl Write for RecordedInput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A PTY whose output never stops. A socket stands in for its file descriptor, so that `poll`
/// reports it readable and writable once, as it does a PTY that is never drained.
struct EndlessOutputPty {
    socket: UnixStream,
    _peer: UnixStream,
    reader: EndlessOutput,
    writer: RecordedInput,
}

impl EndlessOutputPty {
    fn new(writer: RecordedInput) -> Self {
        let (socket, mut peer) = UnixStream::pair().expect("socket pair is created");
        peer.write_all(b"y").expect("socket is made readable");
        Self {
            socket,
            _peer: peer,
            reader: EndlessOutput,
            writer,
        }
    }
}

impl EventedReadWrite for EndlessOutputPty {
    type Reader = EndlessOutput;
    type Writer = RecordedInput;

    fn register(&mut self, poll: &mio::Poll, interest: mio::Interest) -> io::Result<()> {
        poll.registry()
            .register(&mut SourceFd(&self.socket.as_raw_fd()), PTY_TOKEN, interest)
    }

    fn reregister(&mut self, poll: &mio::Poll, interest: mio::Interest) -> io::Result<()> {
        poll.registry()
            .reregister(&mut SourceFd(&self.socket.as_raw_fd()), PTY_TOKEN, interest)
    }

    fn deregister(&mut self, poll: &mio::Poll) -> io::Result<()> {
        poll.registry()
            .deregister(&mut SourceFd(&self.socket.as_raw_fd()))
    }

    fn reader(&mut self) -> &mut EndlessOutput {
        &mut self.reader
    }

    fn read_token(&self) -> mio::Token {
        PTY_TOKEN
    }

    fn writer(&mut self) -> &mut RecordedInput {
        &mut self.writer
    }

    fn write_token(&self) -> mio::Token {
        PTY_TOKEN
    }
}

impl EventedPty for EndlessOutputPty {
    fn child_event_token(&self) -> mio::Token {
        SIGNALS_TOKEN
    }

    fn next_child_event(&mut self) -> Option<ChildEvent> {
        None
    }

    fn on_resize(&mut self, _size: &SizeInfo) {}

    fn kill(self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[test]
fn input_and_shutdown_are_handled_while_output_never_stops() {
    let input = RecordedInput::default();
    let (tx, rx) = mio_channel::channel();
    let event_loop = EventLoop::new(
        test_terminal(),
        ChannelEventListener::new_for_test(),
        EndlessOutputPty::new(input.clone()),
        rx,
    );
    let handle = event_loop.spawn();

    assert!(tx.send(Message::Input(Cow::Borrowed(b"\x03"))).is_ok());
    let deadline = Instant::now() + TIMEOUT;
    while input.0.lock().is_empty() {
        assert!(
            Instant::now() < deadline,
            "input was not written while the PTY kept producing output"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(input.0.lock().as_slice(), b"\x03");

    assert!(tx.send(Message::Shutdown).is_ok());
    let (joined_tx, joined_rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = joined_tx.send(handle.join().is_ok());
    });
    assert_eq!(
        joined_rx.recv_timeout(TIMEOUT),
        Ok(true),
        "the event loop did not shut down while the PTY kept producing output"
    );
}
