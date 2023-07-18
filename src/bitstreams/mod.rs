use serde::{Serialize, Deserialize};

#[derive(Serialize)]
pub struct OutputBitStream {
    pub os: Box<[u8]>,
}

pub struct OutputBitStreamBuilder {
    os: Vec<u8>,
    pub written_bits: usize,
    current: u64,
    free: usize,
    temp_buffer: Vec<u8>,
}

impl Default for OutputBitStreamBuilder {
    fn default() -> Self {
        OutputBitStreamBuilder {
            os: Vec::default(),
            written_bits: 0,
            current: 0,
            free: 8,
            temp_buffer: vec![0u8; 128]
        }
    }
}

impl OutputBitStreamBuilder {    
    pub fn build(mut self) -> OutputBitStream {
        self.write(self.current);

        OutputBitStream {
            os: self.os.into_boxed_slice()
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn write(&mut self, b: u64) {
        self.os.push(b as u8);
    }

    #[inline(always)]
    fn write_in_current(&mut self, b: u64, len: u64) -> u64 {
        self.free -= len as usize;
        self.current |= (b & ((1 << len) - 1)) << self.free;

        if self.free == 0 {
            self.write(self.current);
            self.free = 8;
            self.current = 0;
        }

        self.written_bits += len as usize;
        len
    }

    #[inline(always)]
    pub fn write_unary(&mut self, x: u64) -> u64 {
        if x < self.free as u64 {
            return self.write_in_current(1, x + 1);
        }

        let shift = self.free;
        let x = x - shift as u64;

        self.written_bits += shift;
        self.write(self.current);
        self.free = 8;
        self.current = 0;
        
        let mut i = x >> 3;

        self.written_bits += x as usize & 0x7FFFFFF8;
        
        while i != 0 {
            self.write(0);
            i -= 1;
        }

        self.write_in_current(1, (x & 7) + 1);

        x + shift as u64 + 1
    }

    #[inline(always)]
    fn push_bits(&mut self, x: u64, len: u64) -> u64 {
        assert!(len <= 64, "Cannot write {} bits to an integer", len);

        if len <= self.free as u64 {
            return self.write_in_current(x, len);
        }

        let mut x = x;

        let q = (len - self.free as u64) & 7;
        let blocks = (len - self.free as u64) >> 3;
        
        if q != 0 {
            self.temp_buffer[blocks as usize] = x as u8;
            x >>= q;
        }
        
        let mut i = blocks;
        while i != 0 {
            i -= 1;
            self.temp_buffer[i as usize] = x as u8;
            x >>= 8; 
        }

        self.write_in_current(x, self.free as u64);

        for i in 0..blocks {
            self.write(self.temp_buffer[i as usize] as u64);
        }

        self.written_bits += (blocks << 3) as usize;

        if q != 0 {
            self.write_in_current(self.temp_buffer[blocks as usize] as u64, q);
        }

        len
    }

    // It writes the +1 value, e.g. write_gamma(17) --actual--> gamma(18)
    pub fn write_gamma(&mut self, x: u64) -> u64 {
        assert!(x < u64::MAX);
        // if x < MAX_PRECOMPUTED TODO

        let x = x + 1; // Code [0, +inf - 1]
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;

        self.write_unary(msb) + self.push_bits(x, msb)
    }

    // It writes the +1 value, e.g. write_delta(17) --actual--> delta(18)
    pub fn write_delta(&mut self, x: u64) -> u64 {
        assert!(x < u64::MAX);
        // if x < MAX_PRECOMPUTED TODO

        let x =  x + 1; // Code [0, +inf - 1]
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
        self.write_gamma(msb) + self.push_bits(x, msb)
    }

    pub fn write_zeta(&mut self, x: u64, k: u64) -> u64 {
        assert!(x < u64::MAX);
        assert!(k < u64::MAX);

        let x = x + 1;
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
        let h = msb / k;
        let unary = self.write_unary(h);
        let left = 1 << (h * k);
        unary + 
            if x - left < left 
                {self.push_bits(x - left, h * k + k - 1)}
            else 
                {self.push_bits(x, h * k + k)}
    }
}

#[derive(Debug)]
pub struct InputBitStream {
    is: Box<[u8]>,
    position: usize,
    read_bits: usize,
    current: u64,
    fill: usize,
}

impl InputBitStream {
    pub fn new(input_stream: Box<[u8]>) -> Self {
        InputBitStream { 
            is: input_stream, 
            position: 0, 
            read_bits: 0, 
            current: 0, 
            fill: 0 
        }
    }

    fn position(&mut self, pos: u64) {
        assert!(pos >= 0, "Illegal position {}", pos);

        let bit_delta = ((self.position as u64) << 3) - pos;
        if bit_delta >= 0 && bit_delta as usize <= self.fill {
            self.fill = bit_delta as usize;
            return;
        }

        let delta = (pos >> 3) - self.position as u64;

        self.position = pos as usize >> 3;

        let residual = pos & 7;
        if residual != 0 {
            self.current = self.read().unwrap();
            self.fill = (8 - residual) as usize;
        }
    }

    fn read(&mut self) -> Result<u64, ()> {
        if self.position >= self.is.len() {
            return Err(());
        }

        self.position += 1;
        Ok(self.is[self.position - 1] as u64)
    }

    fn refill(&mut self) -> usize {
        assert!(self.fill < 16);
        
        if let Ok(read) = self.read() {
            self.current = (self.current << 8) | read;
            self.fill += 8;
        }
        if let Ok(read) = self.read() {
            self.current = (self.current << 8) | read;
            self.fill += 8;
        }

        self.fill
    }

    fn read_from_current(&mut self, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }

        if self.fill == 0 {
            self.current = self.read().unwrap();
            self.fill = 8;
        }

        assert!(len as usize <= self.fill);

        self.read_bits += len as usize;

        self.fill -= len as usize;
        self.current >> self.fill & ((1 << len) - 1)
    }

    fn read_int(&mut self, len: u64) -> u64 {
        assert!(len < 64);
        
        if self.fill < 16 {
            self.refill();
        }

        if len as usize <= self.fill {
            return self.read_from_current(len);
        }

        let mut len = len - self.fill as u64;
        
        let mut x = self.read_from_current(self.fill as u64);

        let mut i = len >> 3;

        while i != 0 {
            x = x << 8 | self.read().unwrap();
            i -= 1;
        }

        self.read_bits += len as usize & !7;

        len &= 7;

        (x << len) | self.read_from_current(len)
    }

    fn read_unary(&mut self) -> u64 {
        assert!(self.fill < 64);

        if self.fill < 16 {
            self.refill();
        }

        let mut x = u32::leading_zeros((self.current as u32) << (32 - self.fill));
        if x < self.fill as u32{
            self.read_bits += x as usize + 1;
            self.fill -= x as usize + 1;
            return x as u64;
        }

        x = self.fill as u32;
        let mut read = self.read();

        if read.is_ok() {
            self.current = read.unwrap();
            while self.current == 0 && read.is_ok() {
                x += 8;
                read = self.read();
                if let Ok(r) = read {
                    self.current = r;
                }
            }
        }

        self.fill = (63 - u64::leading_zeros(self.current)) as usize;
        x += 7 - self.fill as u32;
        self.read_bits += x as usize + 1;
        x as u64
    }

    pub fn read_gamma(&mut self) -> u64 {
        let msb = self.read_unary();
        ((1 << msb) | self.read_int(msb)) - 1
    }

    pub fn read_delta(&mut self) -> u64 {
        let msb = self.read_gamma();
        ((1 << msb) | self.read_int(msb)) - 1
    }

    pub fn read_zeta(&mut self, k: u64) -> u64 {
        assert!(k >= 1);

        let unary = self.read_unary();
        let left = 1 << (unary * k);
        let m = self.read_int(unary * k + k - 1);
        if m < left {m + left - 1} else {(m << 1) + self.read_from_current(1) - 1}
    }
}