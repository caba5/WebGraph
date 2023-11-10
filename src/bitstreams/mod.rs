use std::rc::Rc;

pub mod tables;

pub struct BinarySequence {
    pub os: Box<[u8]>,
}

pub struct BinaryWriter {
    os: Vec<u8>,
    pub written_bits: usize,
    pub current: u64,
    pub free: usize,
    temp_buffer: Vec<u8>,
}

impl Default for BinaryWriter {
    fn default() -> Self {
        BinaryWriter {
            os: Vec::default(),
            written_bits: 0,
            current: 0,
            free: 8,
            temp_buffer: vec![0u8; 128]
        }
    }
}

impl BinaryWriter {
    pub fn build(mut self) -> BinarySequence {
        if self.free < 8 {
            self.write(self.current);
        }
        
        BinarySequence {
            os: self.os.into_boxed_slice()
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub(crate) fn write(&mut self, b: u64) {
        self.os.push(b as u8);
    }

    #[inline(always)]
    pub(crate) fn write_in_current(&mut self, b: u64, len: u64) -> u64 {
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
    pub fn push_bits(&mut self, x: u64, len: u64) -> u64 {
        debug_assert!(len <= 64, "Cannot write {} bits to an integer", len);

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
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct BinaryReader {
    pub is: Rc<[u8]>,
    pub position: usize,
    pub read_bits: usize,
    pub current: u64,
    pub fill: usize,
}

impl Default for BinaryReader {
    fn default() -> Self {
        Self { 
            is: Rc::new([]), 
            position: Default::default(), 
            read_bits: Default::default(), 
            current: Default::default(), 
            fill: Default::default()
        }
    }
}

impl BinaryReader {
    pub fn new(input_stream: Rc<[u8]>) -> Self {
        BinaryReader { 
            is: input_stream, 
            ..Default::default()
        }
    }

    #[inline(always)]
    pub fn position(&mut self, pos: u64) {
        let bit_delta = ((self.position as i64) << 3) - pos as i64;
        if bit_delta >= 0 && bit_delta as usize <= self.fill {
            self.fill = bit_delta as usize;
            return;
        }

        self.fill = 0;
        self.position = pos as usize >> 3;

        let residual = pos & 7;

        if residual != 0 {
            self.current = self.read().unwrap();
            self.fill = (8 - residual) as usize;
        }
    }

    #[inline(always)]
    pub fn get_position(&self) -> usize {
        (self.position << 3) - self.fill
    }

    #[inline(always)]
    pub(crate) fn read(&mut self) -> Result<u64, ()> {
        if self.position >= self.is.len() {
            return Err(());
        }

        self.position += 1;
        Ok(self.is[self.position - 1] as u64)
    }

    #[inline(always)]
    pub(crate) fn refill(&mut self) -> usize {
        debug_assert!(self.fill < 16);
        
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

    #[inline(always)]
    pub(crate) fn read_from_current(&mut self, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }

        if self.fill == 0 {
            self.current = self.read().unwrap();
            self.fill = 8;
        }

        debug_assert!(len as usize <= self.fill);

        self.read_bits += len as usize;

        self.fill -= len as usize;
        self.current >> self.fill & ((1 << len) - 1)
    }

    #[inline(always)]
    pub fn read_int(&mut self, len: u64) -> u64 {
        debug_assert!(len < 64);
        
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
}

#[cfg(test)]
mod tests;