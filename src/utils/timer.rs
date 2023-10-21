use std::time::Instant;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Timer {
    pub total_time: u128,
    pub curr: Instant,
}

impl Default for Timer {
    fn default() -> Self {
        Self { total_time: 0, curr: Instant::now() }
    }
}

impl Timer {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn start(&mut self) {
        self.curr = Instant::now();
    }

    #[inline(always)]
    pub fn stop(&mut self) -> u128 {
        let diff = self.curr.elapsed().as_nanos();
        self.total_time += diff;
        diff
    }
}

#[test]
fn test_timer() {
    use std::{thread, time};

    let mut timer = Timer::new();

    thread::sleep(time::Duration::from_nanos(2_000_000_000));

    timer.stop();

    assert!(timer.total_time >= 2_000_000_000);
}