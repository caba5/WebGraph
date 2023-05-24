use std::iter;

pub struct BitStream {
    buffer_size: u32,
    // input_stream: &'a dyn Iterator<Item = u32>,
    read_bits: u32,  // Number of read bits
    current: u32,  // Current bit buffer: the lowest 'fill' bits represent the current content
    buffer: Vec<u8>,  // Stream buffer
    fill: u32,  // Current number of bits in the bit buffer ('current')
    pos: usize,  // Current position in the byte buffer ('buffer')
    avail: u32,  // Current number of available bytes in the byte buffer 
    position: usize,  // Current position of the first byte in the byte buffer
}

impl BitStream {
    pub fn new(byte_arr: &[u8]) -> BitStream {
        if byte_arr.len() > 0 {
            return BitStream {
                buffer_size: 8 * 1024,
                // input_stream: iter::empty::<u32>(),
                read_bits: 0,
                current: 0,
                buffer: Vec::from(byte_arr),
                fill: 0,
                pos: 0,
                avail: byte_arr.len() as u32,
                position: 0
            };
        }
        BitStream { // TODO: is it necessary?
            buffer_size: 8 * 1024,
            read_bits: 0,
            current: 0,
            buffer: Vec::new(),
            fill: 0,
            pos: 0,
            avail: 0,
            position: 0
        }
        
    }

    /// Reads the next byte from the stream.
    pub fn read(&self) -> u32 {
        if self.avail == 0 {
            self.position += self.pos;
            self.pos = 0;
        }

        self.avail -= 1;
        let read_val = self.buffer[self.pos] & 0xFF;
        self.pos += 1;
        read_val as u32
    }

    /// Feeds 16 more bits into [current](BitStream::current).
    pub fn refill(&self) -> u32 {
        assert!(self.fill < 16);

        if self.avail > 1 {
            self.avail -= 2;
            self.current = self.current << 16 | ((self.buffer[self.pos] & 0xFF) << 8 | self.buffer[self.pos + 1] & 0xFF) as u32;
            self.pos += 2;
            self.fill += 16;
            return self.fill;
        }

        self.current = (self.current << 8) | self.read();
        self.fill += 8;
        self.current = (self.current << 8) | self.read();
        self.fill += 8;

        self.fill
    }

    // TODO: https://github.com/vigna/dsiutils/blob/master/src/it/unimi/dsi/io/InputBitStream.java#L501
}