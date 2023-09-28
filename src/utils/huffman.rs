use std::{collections::{HashMap, BTreeMap, BinaryHeap, BTreeSet}, rc::Rc, borrow::BorrowMut, cell::RefCell, cmp::Reverse};

use crate::{bitstreams::{BinaryWriterBuilder, BinaryReader}, EncodingType};

#[derive(Default, Debug)]
pub struct HeapNode {
    key: usize,
    data: usize,
    left: Option<Rc<HeapNode>>,
    right: Option<Rc<HeapNode>>,
}

impl HeapNode {
    fn new(key: usize, data: usize) -> Self {
        Self { key, data, ..Default::default() }
    }

    fn set_left_child(&mut self, child: Option<Rc<HeapNode>>) {
        self.left = child;
    }

    fn set_right_child(&mut self, child: Option<Rc<HeapNode>>) {
        self.right = child;
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.data == other.data {
            return self.key.cmp(&other.key);
        }
        self.data.cmp(&other.data)
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.data == other.data {
            return Some(self.key.cmp(&other.key));
        }
        Some(self.data.cmp(&other.data))
    }
}

impl Eq for HeapNode {}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

#[derive(Default)]
pub struct HuffmanEncoder {
    data: Box<[(usize, usize, usize)]>,
    freq_tree: BinaryHeap<Reverse<Rc<HeapNode>>>,
    freq_map: HashMap<usize, usize>,
    code_len_map: BTreeMap<usize, BTreeSet<usize>>,
    /// Contains the original data as keys and tuples (code_length, code) as values
    canonical_code_map: HashMap<usize, (usize, usize)>,
    code_bits: usize,
    num_values_bits: usize,
}

impl HuffmanEncoder {
    pub fn new(v: Box<[(usize, usize, usize)]>, code_bits: usize, num_values_bits: usize) -> Self {
        Self { data: v, code_bits, num_values_bits, ..Default::default() }
    }

    pub fn get_frequency(&mut self) {
        for &x in self.data.iter() {
            self.freq_map.entry(x.0).and_modify(|freq| *freq += 1).or_insert(1);
        }
    }

    pub fn create_huffman_tree(&mut self) -> Option<Rc<HeapNode>>{
        for (&k, &v) in self.freq_map.iter() {
            self.freq_tree.push(Reverse(Rc::new(HeapNode::new(k, v))));
        }

        let mut root = 
            if self.freq_tree.len() == 1 {
                Some(self.freq_tree.peek().unwrap().0.clone())
            } else {
                None
            };

        while self.freq_tree.len() > 1 {
            let node1 = self.freq_tree.pop().unwrap().clone();
            let node2 = self.freq_tree.pop().unwrap().clone();

            let mut new_node = HeapNode::new(0, node1.0.data + node2.0.data);
            new_node.set_left_child(Some(node1.0));
            new_node.set_right_child(Some(node2.0));

            let new_node = Rc::new(new_node);
            
            root = Some(new_node.clone());

            self.freq_tree.push(Reverse(new_node));
        }

        root
    }

    pub fn get_code_len(&mut self, root: Option<Rc<HeapNode>>, len: usize) {
        if let Some(root) = root {
            if root.left.is_none() && root.right.is_none() {
                self.code_len_map.entry(len)
                                    .and_modify(|int_set| { int_set.insert(root.key); })
                                    .or_insert_with(|| { let mut hs = BTreeSet::new(); hs.insert(root.key); hs });
                return;                
            }

            self.get_code_len(root.left.clone(), len + 1);
            self.get_code_len(root.right.clone(), len + 1);
        }
    }

    pub fn generate_canonical_code(&mut self) {
        let mut prev_len = 0;
        let mut curr_val = 0;
        let mut canonical_len = 0;

        for (&len, int_set) in self.code_len_map.iter() {
            canonical_len += len - prev_len;
            curr_val <<= len - prev_len;
            for &i in int_set.iter() {
                self.canonical_code_map.insert(i, (if canonical_len == 0 {1} else {canonical_len}, curr_val));
                curr_val += 1;
            }
            prev_len = len;
        }
    }

    pub fn write_header(&self, writer: &mut BinaryWriterBuilder) {
        let mut ordered_codes = Vec::with_capacity(self.canonical_code_map.len());

        for (&k, &v) in self.canonical_code_map.iter() {
            ordered_codes.push((k, v));
        }

        // Order by increasing code length
        ordered_codes.sort_by(|first, second| if first.1.0 == second.1.0 {first.0.cmp(&second.0)} else {first.1.0.cmp(&second.1.0)});

        let max_len = ordered_codes.last().unwrap().1.0;
        let int_num = ordered_codes.len();

        writer.push_bits(max_len as u64, 8); // The first 8 bits will represent the maximum code_length

        let mut search_index = 0;

        for i in 1..max_len + 1 {
            let mut count = 0;
            while search_index < ordered_codes.len() && i == ordered_codes[search_index].1.0 {
                count += 1;
                search_index += 1;
            }
            writer.push_bits(count, 8);
        }

        writer.push_bits(int_num as u64, 8); // The total number of unique zuckerli-encoded values // TODO: check correctness for large graphs
        for x in ordered_codes.iter() {
            writer.push_bits(x.0 as u64, self.code_bits as u64);
        }

        writer.push_bits(self.data.len() as u64, self.num_values_bits as u64);
    }

    pub fn write_body(&self, writer: &mut BinaryWriterBuilder) {
        for int in self.data.iter() {
            let length = self.canonical_code_map.get(&int.0).unwrap().0;
            let code = self.canonical_code_map.get(&int.0).unwrap().1;

            writer.push_bits(code as u64, length as u64);
            writer.push_bits(int.2 as u64, int.1 as u64);
        }
    }

    pub fn write(&self, writer: &mut BinaryWriterBuilder, value: usize) {
        let (x, zuck_t_len, zuck_t) = encode(value, 4, 2, 1);

        let item = self.canonical_code_map.get(&x);

        debug_assert!(item.is_some(), "The provided value {} has not been encoded", x);

        let (length, code) = item.unwrap();

        writer.push_bits(*code as u64, *length as u64);
        writer.push_bits(zuck_t as u64, zuck_t_len as u64);
    }
        
    pub fn build_huffman_zuck(data: &Vec<usize>) -> Self {
        let mut transformed_data = Vec::with_capacity(data.len());

        let mut max_data = 0;
        let mut max_len = 0;
        let mut max_t = 0;

        for &x in data.iter() {
            let zuck_triple = encode(x, 4, 2, 1);
            transformed_data.push(zuck_triple);
            max_data = zuck_triple.0.max(max_data);
            max_len = zuck_triple.1.max(max_len);
            max_t = zuck_triple.2.max(max_t);
        }

        let bits_data = HuffmanEncoder::get_minimum_amount_bits(max_data);
        let bits_values = HuffmanEncoder::get_minimum_amount_bits(data.len());

        println!("Bits needed for max data {}: {}", max_data, bits_data);
        println!("Bits needed for the number of values {}: {}", data.len(), bits_values);

        let mut huff = HuffmanEncoder::new(transformed_data.into_boxed_slice(), bits_data, bits_values);
        
        huff.get_frequency();
        let root = huff.create_huffman_tree();
        huff.get_code_len(root, 0);
        huff.generate_canonical_code();

        huff
    }

    pub fn write_huffman_zuck(encoder: &HuffmanEncoder, writer: &mut BinaryWriterBuilder) {
        encoder.write_header(writer);
        encoder.write_body(writer);
    }

    fn get_minimum_amount_bits(x: usize) -> usize {
        let mut amount = 1;
        let mut curr = 1;

        while curr <= x {
            curr <<= 1;
            amount += 1;
        }

        amount - 1
    }
}

pub struct HeapNodeDecoder {
    key: usize,
    data: usize,
    left: Option<Rc<RefCell<HeapNodeDecoder>>>,
    right: Option<Rc<RefCell<HeapNodeDecoder>>>,
    intermediate: bool,
}

impl HeapNodeDecoder {
    pub fn new(key: usize, data: usize, intermediate: bool) -> Self {
        Self { key, data, left: None, right: None, intermediate }
    }

    pub fn set_left_child(&mut self, left: Option<Rc<RefCell<HeapNodeDecoder>>>) {
        self.left = left;
    }

    pub fn set_right_child(&mut self, right: Option<Rc<RefCell<HeapNodeDecoder>>>) {
        self.right = right;
    }
}

#[derive(Default)]
pub struct HuffmanDecoder {
    reader: Rc<RefCell<BinaryReader>>,
    canonical_tree_root: Option<Rc<RefCell<HeapNodeDecoder>>>,
    output_vec: Vec<usize>,
    code_bits: usize,
    values_bits: usize,
}

impl HuffmanDecoder {
    pub fn new(reader: Rc<RefCell<BinaryReader>>, code_bits: usize, values_bits: usize) -> Self {
       Self { 
        reader,
        canonical_tree_root: None,
        output_vec: Vec::new(),
        code_bits,
        values_bits
        }
    }

    pub fn read_header(&mut self) {
        let max_len = (*self.reader).borrow_mut().read_int(8) as usize;
        let mut length = Vec::with_capacity(max_len);
        for _ in 0..max_len {
            length.push((*self.reader).borrow_mut().read_int(8) as usize);
        }

        let number_of_ints = (*self.reader).borrow_mut().read_int(8) as usize;
        let mut symbols = Vec::with_capacity(number_of_ints);
        for _ in 0..number_of_ints {
            symbols.push((*self.reader).borrow_mut().read_int(self.code_bits as u64) as usize);
        }

        let root = HeapNodeDecoder::new(0, 0, true);
        let root = Rc::new(RefCell::new(root));

        let mut symbols_index = 0;
        let mut curr_code;
        let mut previous_code = -1;
        let mut previous_len = 0;

        for (i, &len) in length.iter().enumerate() {
            if len > 0 {
                for _ in 0..len {
                    let int = symbols[symbols_index];

                    curr_code = ((previous_code + 1) as usize) << (i - previous_len);

                    self.add_to_code_tree(root.clone(), int, i + 1, curr_code);

                    symbols_index += 1;
                    previous_code = curr_code as i64;
                    previous_len = i;
                }
            }
        }

        self.canonical_tree_root = Some(root)
    }

    fn add_to_code_tree(&mut self, root: Rc<RefCell<HeapNodeDecoder>>, int: usize, len: usize, code: usize) {
        let mut current = Some(root);
        let mut _next = None;

        let mut i = len;
        while i > 1 {
            let bit = if ((1 << (i - 1)) & code) > 0 {1} else {0};

            if bit == 1 {
                if current.as_ref().unwrap().borrow().right.is_some() {
                    _next = current.as_ref().unwrap().borrow().right.clone();
                } else {
                    let n = HeapNodeDecoder::new(0, 0, true);
                    let n = Rc::new(RefCell::new(n));
                    
                    let mut g = (**current.as_mut().unwrap().borrow_mut()).borrow_mut();
                    g.set_right_child(Some(n.clone()));

                    _next = Some(n);
                }
            } else if current.as_ref().unwrap().borrow().left.is_some() {
                _next = current.as_ref().unwrap().borrow().left.clone();
            } else {
                let n = HeapNodeDecoder::new(0, 0, true);
                let n = Rc::new(RefCell::new(n));
                
                let mut g = (**current.as_mut().unwrap().borrow_mut()).borrow_mut();
                g.set_left_child(Some(n.clone()));

                _next = Some(n);
            }

            current = _next;

            i -= 1;
        }

        let bit = if 1 & code > 0 {1} else {0};

        if bit == 1 {
            let n = HeapNodeDecoder::new(int, 0, false);
            let n = Rc::new(RefCell::new(n));

            (**current.as_mut().unwrap().borrow_mut()).borrow_mut().set_right_child(Some(n));
        } else {
            let n = HeapNodeDecoder::new(int, 0, false);
            let n = Rc::new(RefCell::new(n));

            (**current.as_mut().unwrap().borrow_mut()).borrow_mut().set_left_child(Some(n));
        }
    }

    pub fn read_body(&mut self) {
        let to_read = (*self.reader).borrow_mut().read_int(self.values_bits as u64);

        self.output_vec = Vec::with_capacity(to_read as usize);
        
        let mut curr_node = Some(self.canonical_tree_root.as_ref().unwrap().clone());
        let mut i = 0;

        while i < to_read && curr_node.is_some() {
            if !curr_node.as_ref().unwrap().borrow().intermediate {                             
                let decoded = decode( curr_node.as_ref().unwrap().borrow().key, &mut (*self.reader).borrow_mut(), 4, 2, 1);
                self.output_vec.push(decoded);

                curr_node = Some(self.canonical_tree_root.as_ref().unwrap().clone());

                i += 1;
            } else {
                let bit = (*self.reader).borrow_mut().read_int(1);

                let next;

                if bit == 1 {
                    if curr_node.as_ref().unwrap().borrow().right.is_some() {
                        next = Some(curr_node.as_ref().unwrap().borrow().right.as_ref().unwrap().clone());
                    } else {
                        next = None;
                    }
                } else if curr_node.as_ref().unwrap().borrow().left.is_some() {
                    next = Some(curr_node.as_ref().unwrap().borrow().left.as_ref().unwrap().clone())
                } else {
                    next = None;
                };

                curr_node = next;
            }
        }
    }

    pub fn decode_huffman_zuck(reader: BinaryReader, code_bits: usize, values_bits: usize) -> Vec<usize> {        
        let mut huff_decoder = HuffmanDecoder::new(Rc::new(RefCell::new(reader)), code_bits, values_bits);

        huff_decoder.read_header();

        huff_decoder.read_body();

        huff_decoder.output_vec
    }
}

#[inline(always)]
pub fn decode(token: usize, reader: &mut BinaryReader, k: usize, msb_in_token /* (i) */: usize, lsb_in_token /* (j) */: usize) -> usize { // 4 2 1
    assert!(k >= lsb_in_token + msb_in_token);
    let split_token = 1 << k; // 2^k

    if token < split_token {
        return token;
    }

    let nbits = k - (msb_in_token + lsb_in_token) +
                        ((token - split_token) >> (msb_in_token + lsb_in_token));
    let low = token & ((1 << lsb_in_token) - 1);
    let token = token >> lsb_in_token;

    let bits = reader.read_int(nbits as u64) as usize;
    (((((1 << msb_in_token) | (token & ((1 << msb_in_token) - 1)))
        << nbits) |
        bits)
        << lsb_in_token) |
        low
}

#[inline(always)]
pub fn encode(value: usize, k: usize, msb_in_token /* (i) */: usize, lsb_in_token /* (j) */: usize) -> (usize, usize, usize) {
    assert!(k >= lsb_in_token + msb_in_token);
    let split_token = 1 << k; // 2^k

    if value < split_token {
        return (value, 0, 0);
    }

    let n = (usize::BITS - 1 - value.leading_zeros()) as usize;
    let m = value - (1 << n);

    let s = split_token +                           // 2^k +
        ((n - k) << (msb_in_token + lsb_in_token)) +       // (p - k - 1) * 2^(i+j) +
        ((m >> (n -msb_in_token)) << lsb_in_token) +       // m * 2^j +
        (m & ((1 << lsb_in_token) - 1));                   // l
    let t_len = n - msb_in_token - lsb_in_token;
    let t = (value >> lsb_in_token) & ((1 << t_len) - 1);
    (s, t_len, t)
}

#[test]
fn test_huffman_zuck_large_sequence() {
    let v = Vec::from_iter(0..65534);

    let huff = HuffmanEncoder::build_huffman_zuck(&v);

    let mut binary_writer = BinaryWriterBuilder::new();

    HuffmanEncoder::write_huffman_zuck(&huff, &mut binary_writer);

    let code_bits = huff.code_bits;
    let values_bits = huff.num_values_bits;
    let out = binary_writer.build().os;

    let reader = BinaryReader::new(out.into());

    let result = HuffmanDecoder::decode_huffman_zuck(reader, code_bits, values_bits);

    assert_eq!(result, v);
}

#[test]
fn test_huffman_zuck_single_number() {
    let v = vec![1, 1, 1, 1, 1, 1, 1];

    let huff = HuffmanEncoder::build_huffman_zuck(&v);

    let mut binary_writer = BinaryWriterBuilder::new();

    HuffmanEncoder::write_huffman_zuck(&huff, &mut binary_writer);

    let code_bits = huff.code_bits;
    let values_bits = huff.num_values_bits;
    let out = binary_writer.build().os;

    let reader = BinaryReader::new(out.into());

    let result = HuffmanDecoder::decode_huffman_zuck(reader, code_bits, values_bits);

    assert_eq!(result, v);
}

#[test]
fn test_huffman_zuck_normal_sequence() {
    let v = vec![10000, 20000, 65535, 65535, 30000, 1, 1, 20000, 3, 3, 100, 200, 65535, 65535, 1];

    let huff = HuffmanEncoder::build_huffman_zuck(&v);

    let mut binary_writer = BinaryWriterBuilder::new();

    HuffmanEncoder::write_huffman_zuck(&huff, &mut binary_writer);

    let code_bits = huff.code_bits;
    let values_bits = huff.num_values_bits;
    let out = binary_writer.build().os;

    let reader = BinaryReader::new(out.into());

    let result = HuffmanDecoder::decode_huffman_zuck(reader, code_bits, values_bits);

    assert_eq!(result, v);
}

#[test]
fn test_huffman_iterative_write() {
    let v = vec![10000, 20000, 65535, 65535, 30000, 1, 1, 20000, 3, 3, 100, 200, 65535, 65535, 1];
    
    let huff = HuffmanEncoder::build_huffman_zuck(&v);

    let mut binary_writer = BinaryWriterBuilder::new();

    huff.write_header(&mut binary_writer);

    for &x in v.iter() {
        huff.write(&mut binary_writer, x);
    }

    let code_bits = huff.code_bits;
    let values_bits = huff.num_values_bits;
    let out = binary_writer.build().os;

    let reader = BinaryReader::new(out.into());

    let result = HuffmanDecoder::decode_huffman_zuck(reader, code_bits, values_bits);

    assert_eq!(result, v);
}

pub struct HuffmanDecoderIterator {
    decoder: HuffmanDecoder,
    decoded: bool,
    i: usize,
    len: usize,
}

impl Default for HuffmanDecoderIterator {
    fn default() -> Self {
        Self { decoder: Default::default(), decoded: Default::default(), i: Default::default(), len: Default::default() }
    }
}

impl HuffmanDecoderIterator {
    pub fn new(reader: Rc<RefCell<BinaryReader>>, code_bits: usize, t_len_bits: usize) -> Self {
        Self { decoder: HuffmanDecoder::new(reader, code_bits, t_len_bits), decoded: false, i: 0, len: 0 }
    }
}

impl Iterator for HuffmanDecoderIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.decoded {
            self.decoder.read_header();
            self.decoder.read_body();
            self.decoded = true;
            self.len = self.decoder.output_vec.len();
        }

        if self.i >= self.len {
            return None;
        }

        Some(self.decoder.output_vec[self.i])
    }
}

pub trait Huffman {
    fn read_next(decoder_it: &mut HuffmanDecoderIterator) -> u64;

    fn to_encoding_type() -> EncodingType;
}

pub struct Huff;

impl Huffman for Huff {
    #[inline(always)]
    fn read_next(decoder_it: &mut HuffmanDecoderIterator) -> u64 {
        decoder_it.next().unwrap() as u64
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::HUFFMAN
    }
}