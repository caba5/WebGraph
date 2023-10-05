use std::{collections::{HashMap, BTreeMap, BinaryHeap, BTreeSet}, rc::Rc, borrow::BorrowMut, cell::RefCell, cmp::Reverse};

use crate::{bitstreams::{BinaryWriterBuilder, BinaryReader}, utils::encodings::zuck_encode};

use super::encodings::{K_ZUCK, I_ZUCK, J_ZUCK, zuck_decode};

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
    // Contains the original data as keys and tuples (code_length, code) as values
    canonical_code_map: HashMap<usize, (usize, usize)>,
    /// The number of bits needed to encode the largest datum
    pub code_bits: usize,
    /// The number of bits needed to encode the integer representing the number of encoded values
    pub num_values_bits: usize,
}

impl HuffmanEncoder {
    /// Creates a new `HuffmanEncoder`.
    /// 
    /// # Arguments
    /// 
    /// * `v` - A `Box` of tuples containing Zuckerli coding triples.
    /// * `code_bits` - The number of bits needed to encode the largest datum.
    /// * `num_values_bits` - The number of bits needed to encode the integer representing the number of encoded values.
    pub fn new(v: Box<[(usize, usize, usize)]>, code_bits: usize, num_values_bits: usize) -> Self {
        Self { data: v, code_bits, num_values_bits, ..Default::default() }
    }

    fn get_frequency(&mut self) {
        for &x in self.data.iter() {
            self.freq_map.entry(x.0).and_modify(|freq| *freq += 1).or_insert(1);
        }
    }

    fn create_huffman_tree(&mut self) -> Option<Rc<HeapNode>>{
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

    fn get_code_len(&mut self, root: Option<Rc<HeapNode>>, len: usize) {
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

    fn generate_canonical_code(&mut self) {
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

    /// Writes the canonical Huffman's header onto a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `writer` - The binary stream onto which to write the header.
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

    /// Writes all the encoded integers contained in `self.data` to a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `writer` - The binary stream onto which to write the data.
    pub fn write_body(&self, writer: &mut BinaryWriterBuilder) {
        for int in self.data.iter() {
            let length = self.canonical_code_map.get(&int.0).unwrap().0;
            let code = self.canonical_code_map.get(&int.0).unwrap().1;

            writer.push_bits(code as u64, length as u64);
            writer.push_bits(int.2 as u64, int.1 as u64);
        }
    }

    /// Iteratively encodes and writes a value to a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `writer` - The binary stream onto which to write the encoded value.
    /// * `value` - The integer to encode.
    pub fn write(&self, writer: &mut BinaryWriterBuilder, value: usize) {
        let (x, zuck_t_len, zuck_t) = zuck_encode(value, K_ZUCK, I_ZUCK, J_ZUCK);

        let item = self.canonical_code_map.get(&x);

        debug_assert!(item.is_some(), "The value {} has not been encoded", x);

        let (length, code) = item.unwrap();

        writer.push_bits(*code as u64, *length as u64);
        writer.push_bits(zuck_t as u64, zuck_t_len as u64);
    }
    
    /// Creates the Huffman encoding of the Zuckerli-transformed data.
    /// 
    /// # Arguments
    /// 
    /// * `data` - The data to be encoded.
    pub fn build_huffman_zuck(data: &Vec<usize>) -> Self {
        let mut transformed_data = Vec::with_capacity(data.len());

        let mut max_data = 0;
        let mut max_len = 0;
        let mut max_t = 0;

        for &x in data.iter() {
            let zuck_triple = zuck_encode(x, K_ZUCK, I_ZUCK, J_ZUCK);
            transformed_data.push(zuck_triple);
            max_data = zuck_triple.0.max(max_data);
            max_len = zuck_triple.1.max(max_len);
            max_t = zuck_triple.2.max(max_t);
        }

        let bits_data = HuffmanEncoder::get_minimum_amount_bits(max_data);
        let bits_values = HuffmanEncoder::get_minimum_amount_bits(data.len());

        let mut huff = HuffmanEncoder::new(transformed_data.into_boxed_slice(), bits_data, bits_values);
        
        huff.get_frequency();
        let root = huff.create_huffman_tree();
        huff.get_code_len(root, 0);
        huff.generate_canonical_code();

        huff
    }

    /// Writes the content of an `HuffmanEncoder` to a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `encoder` - The `HuffmanEncoder` containing data and its representations.
    /// * `writer` - The binary writer onto which to write.
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
    num_values_bits: usize,
    to_read: usize,
}

impl HuffmanDecoder {
    /// Creates a HuffmanDecoder.
    /// 
    /// # Arguments
    /// 
    /// * `reader` - A shared `RefCell` containing the binary reader to read.
    /// * `code_bits` - The number of bits needed for the largest datum to decode.
    /// * `num_values_bits` - The number of bits needed to decode the integer representing the total number of values.
    pub fn new(reader: Rc<RefCell<BinaryReader>>, code_bits: usize, num_values_bits: usize) -> Self {
       Self { 
        reader,
        canonical_tree_root: None,
        output_vec: Vec::new(),
        code_bits,
        num_values_bits,
        to_read: 0
        }
    }

    /// Reads the canonical Huffman's header.
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

        self.canonical_tree_root = Some(root);
        self.to_read = (*self.reader).borrow_mut().read_int(self.num_values_bits as u64) as usize;
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

    /// Reads the whole body of the Huffman-encoded sequence, populating the internal `output_vec`.
    pub fn read_body(&mut self) {
        self.output_vec = Vec::with_capacity(self.to_read);
        
        let mut curr_node = Some(self.canonical_tree_root.as_ref().unwrap().clone());
        let mut i = 0;

        while i < self.to_read && curr_node.is_some() {
            if !curr_node.as_ref().unwrap().borrow().intermediate {                             
                let decoded = zuck_decode( curr_node.as_ref().unwrap().borrow().key, &mut (*self.reader).borrow_mut(), K_ZUCK, I_ZUCK, J_ZUCK);
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

    /// Iteratively reads the next Huffman-encoded value.
    /// 
    /// # Arguments
    /// 
    /// * `reader` - The binary reader containing the bits to read.
    #[inline(always)]
    pub fn read(&mut self, reader: &mut BinaryReader) -> usize {
        let mut res = 0;

        let mut curr_node = Some(self.canonical_tree_root.as_ref().unwrap().clone());

        while curr_node.is_some() {
            if !curr_node.as_ref().unwrap().borrow().intermediate {                             
                let decoded = zuck_decode( curr_node.as_ref().unwrap().borrow().key, reader, K_ZUCK, I_ZUCK, J_ZUCK);
                res = decoded;

                curr_node = None;
            } else {
                let bit = reader.read_int(1);

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

        res
    }

    /// Creates a `HuffmanDecoder` instance, decodes the header, decodes the body, and returns all the read values.
    /// 
    /// # Arguments
    /// 
    /// * `reader` - The binary reader containing the bits to read.
    /// * `code_bits` - The number of bits needed to represent the largest encoded datum.
    /// * `values_bits` - The number of bits needed to represent the number of encoded values.
    pub fn decode_huffman_zuck(reader: BinaryReader, code_bits: usize, values_bits: usize) -> Vec<usize> {        
        let mut huff_decoder = HuffmanDecoder::new(Rc::new(RefCell::new(reader)), code_bits, values_bits);

        huff_decoder.read_header();

        huff_decoder.read_body();

        huff_decoder.output_vec
    }
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

#[test]
fn test_huffman_iterative_read() {
    let v = vec![10000, 20000, 65535, 65535, 30000, 1, 1, 20000, 3, 3, 100, 200, 65535, 65535, 1];

    let huff = HuffmanEncoder::build_huffman_zuck(&v);

    let mut binary_writer = BinaryWriterBuilder::new();

    HuffmanEncoder::write_huffman_zuck(&huff, &mut binary_writer);

    let code_bits = huff.code_bits;
    let values_bits = huff.num_values_bits;
    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff = HuffmanDecoder::new(reader.clone(), code_bits, values_bits);
    huff.read_header();

    let mut result = Vec::new();

    for _ in 0..v.len() {
        result.push(huff.read(&mut (*reader).borrow_mut()));
    }

    assert_eq!(result, v);
}

#[test]
fn test_huffman_interleaved() {
    let v1 = vec![10000, 20000, 65535, 65535, 30000];
    let v2 = vec![1, 1, 20000, 3, 3];
    let v3 = vec![100, 200, 65535, 65535, 1];

    let huff1 = HuffmanEncoder::build_huffman_zuck(&v1);
    let huff2 = HuffmanEncoder::build_huffman_zuck(&v2);
    let huff3 = HuffmanEncoder::build_huffman_zuck(&v3);

    let mut binary_writer = BinaryWriterBuilder::new();

    huff1.write_header(&mut binary_writer);
    huff2.write_header(&mut binary_writer);
    huff3.write_header(&mut binary_writer);

    for i in 0..5 {
        huff1.write(&mut binary_writer, v1[i]);
        huff2.write(&mut binary_writer, v2[i]);
        huff3.write(&mut binary_writer, v3[i]);
    }

    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff1 = HuffmanDecoder::new(reader.clone(), huff1.code_bits, huff1.num_values_bits);
    let mut huff2 = HuffmanDecoder::new(reader.clone(), huff2.code_bits, huff2.num_values_bits);
    let mut huff3 = HuffmanDecoder::new(reader.clone(), huff3.code_bits, huff3.num_values_bits);

    huff1.read_header();
    huff2.read_header();
    huff3.read_header();

    let mut result1 = Vec::new();
    let mut result2 = Vec::new();
    let mut result3 = Vec::new();

    for _ in 0..5 {
        result1.push(huff1.read(&mut (*reader).borrow_mut()));
        result2.push(huff2.read(&mut (*reader).borrow_mut()));
        result3.push(huff3.read(&mut (*reader).borrow_mut()));
    }

    assert_eq!(result1, v1);
    assert_eq!(result2, v2);
    assert_eq!(result3, v3);
}

#[test]
fn test_huffman_interleaved_multiple_encodings() {
    let v1 = vec![10000, 20000, 65535, 65535, 30000];
    let v2 = vec![1, 1, 20000, 3, 3];
    let v3 = vec![100, 200, 65535, 65535, 1];

    let huff1 = HuffmanEncoder::build_huffman_zuck(&v1);
    let huff2 = HuffmanEncoder::build_huffman_zuck(&v2);
    let huff3 = HuffmanEncoder::build_huffman_zuck(&v3);

    let mut binary_writer = BinaryWriterBuilder::new();

    huff1.write_header(&mut binary_writer);
    huff2.write_header(&mut binary_writer);
    huff3.write_header(&mut binary_writer);

    for i in 0..5 {
        huff1.write(&mut binary_writer, v1[i]);
        huff2.write(&mut binary_writer, v2[i]);
        huff3.write(&mut binary_writer, v3[i]);
    }

    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff1 = HuffmanDecoder::new(reader.clone(), huff1.code_bits, huff1.num_values_bits);
    let mut huff2 = HuffmanDecoder::new(reader.clone(), huff2.code_bits, huff2.num_values_bits);
    let mut huff3 = HuffmanDecoder::new(reader.clone(), huff3.code_bits, huff3.num_values_bits);

    huff1.read_header();
    huff2.read_header();
    huff3.read_header();

    let mut result1 = Vec::new();
    let mut result2 = Vec::new();
    let mut result3 = Vec::new();

    for _ in 0..5 {
        result1.push(huff1.read(&mut (*reader).borrow_mut()));
        result2.push(huff2.read(&mut (*reader).borrow_mut()));
        result3.push(huff3.read(&mut (*reader).borrow_mut()));
    }

    assert_eq!(result1, v1);
    assert_eq!(result2, v2);
    assert_eq!(result3, v3);
}