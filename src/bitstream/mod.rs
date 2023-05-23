pub struct BitStream<'a> {
    buffer_size: u32,
    input_stream: &'a dyn Iterator<Item = u32>,
    read_bits: u32,
    current: u32,
    buffer: Vec<u8>,
    fill: u32,
    pos: usize,
    avail: u32,
    position: u32,
}

impl<'a> BitStream<'a> {
    pub fn new(input_stream: &dyn Iterator<Item = u32>, buffer_size: u32) -> BitStream {
        BitStream {
            buffer_size: if buffer_size > 0 {buffer_size} else {0},
            input_stream,
            read_bits: 0,
            current: 0,
            buffer: Vec::with_capacity(buffer_size as usize),
            fill: 0,
            pos: 0,
            avail: 0,
            position: 0
        }
    }

    pub fn read(&self) -> u32 {
        todo!() // https://github.com/vigna/dsiutils/blob/master/src/it/unimi/dsi/io/InputBitStream.java
    }

    pub fn refill(&self) -> u32 {
        assert!(self.fill < 16);

        if self.avail > 1 {
            self.avail -= 2;
            self.current <<= 16 | (self.buffer[self.pos] & 0xFF) << 8;
            self.pos += 1;
            self.current |= self.buffer[self.pos] as u32 & 0xFF;
            self.pos += 1;
        }

        self.current = (self.current << 8) | self.read();
        self.fill += 8;
        self.current = (self.current << 8) | self.read();
        self.fill += 8;

        self.fill
    }
}