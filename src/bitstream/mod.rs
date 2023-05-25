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
            return Self {
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
        Self { // TODO: is it necessary?
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

    /// Reads bits from the bit buffer, possibly refilling it.
    fn read_from_current(&self, len: u32) -> u32 {
        if len == 0 {
            return 0;
        }

        if self.fill == 0 {
            self.current = self.read();
            self.fill = 8;
        }

        assert!(len <= self.fill);

        self.read_bits += len;

        self.fill -= len;

        self.current >> self.fill & (1 << len) - 1
    }

    pub fn read_k_bits(&self, len: u32) -> &[u8] {
        assert!(self.fill < 32);
        
        let mut bits: Vec<u8> = Vec::new();

        if len <= self.fill {
            if len <= 8 {
                bits.push((self.read_from_current(len) << 8 - len) as u8);
                return bits.as_slice();
            }
            if len <= 16 {
                bits.push(self.read_from_current(len) as u8);
                bits.push((self.read_from_current(len - 8) << 16 - len) as u8);
                return bits.as_slice();
            }
            if len <= 24 {
                bits.push(self.read_from_current(8) as u8);
                bits.push(self.read_from_current(8) as u8);
                bits.push((self.read_from_current(len - 16) << 24 - len) as u8);
                return bits.as_slice();
            }
            if len <= 32 {
                bits.push(self.read_from_current(8) as u8);
                bits.push(self.read_from_current(8) as u8);
                bits.push(self.read_from_current(8) as u8);
                bits.push((self.read_from_current(len - 24) << 32 - len) as u8);
                return bits.as_slice();
            }
        } else {
            
        }

        todo!()
    }

    // TODO: https://github.com/vigna/dsiutils/blob/master/src/it/unimi/dsi/io/InputBitStream.java#L501
}