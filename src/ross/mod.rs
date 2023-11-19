use crate::{bitstreams::{BinaryWriter, BinaryReader}, utils::encodings::{GammaCode, UniversalCode}};

#[derive(Default)]
pub struct Ross {
    boundaries: Box<[u64]>,
    class_bits: Box<[u64]>,

    pub first_classes: u64,
    pub last_class: u64,
}

impl Ross {
    pub fn new() -> Self {
        Ross::default()
    }

    pub fn init(&mut self) { // TODO
        self.boundaries = Box::new([0, 4, 20, 84, 148, 276, 532, 788, 916, 1044, 1172, 1428, 1684, 1940, 2196, 2452, 2549]);

        let mut bits = Vec::with_capacity(self.boundaries.len() - 1);
        for i in 0..self.boundaries.len() - 1 {
            bits.push(f64::log2((self.boundaries[i + 1] - self.boundaries[i]) as f64).ceil() as u64);
        }

        self.class_bits = bits.into_boxed_slice();
    }

    #[inline(always)]
    pub fn encode(&mut self, value: u64, binary_writer: &mut BinaryWriter) {
        let class = self.boundaries.binary_search(&value).unwrap_or_else(|x| x - 1);
        let delta = value - self.boundaries[class];

        let sz = GammaCode::write_next(binary_writer, class as u64, None);

        // If the found class is the last one, then the delta is encoded in Gamma
        if class >= self.class_bits.len() { self.last_class += sz + GammaCode::write_next(binary_writer, delta, None); } 
        else { self.first_classes += sz + binary_writer.push_bits(delta, self.class_bits[class]); }
    }

    #[inline(always)]
    pub fn decode(&self, binary_reader: &mut BinaryReader) -> u64 {
        let class = GammaCode::read_next(binary_reader, None) as usize;

        if class >= self.class_bits.len() {
            let delta = GammaCode::read_next(binary_reader, None);
            delta + self.boundaries.last().unwrap()
        } else {
            let delta = binary_reader.read_int(self.class_bits[class]);
            self.boundaries[class] + delta
        }
    }
}

#[test]
fn test_write_1() {
    let mut writer = BinaryWriter::new();

    let mut ross = Ross::new();
    ross.init();

    ross.encode(0, &mut writer);
    ross.encode(2, &mut writer);
    ross.encode(3, &mut writer);
    ross.encode(5, &mut writer);
    ross.encode(2849, &mut writer);

    let mut reader = BinaryReader::new(writer.build().os.into());

    assert_eq!(ross.decode(&mut reader), 0);
    assert_eq!(ross.decode(&mut reader), 2);
    assert_eq!(ross.decode(&mut reader), 3);
    assert_eq!(ross.decode(&mut reader), 5);
    assert_eq!(ross.decode(&mut reader), 2849);
}