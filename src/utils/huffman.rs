use std::{collections::{HashMap, BTreeMap, BinaryHeap, BTreeSet}, rc::Rc, borrow::BorrowMut, cell::RefCell, cmp::Reverse};

use crate::{bitstreams::{BinaryWriterBuilder, BinaryReader}, utils::encodings::{zuck_encode, K_ZUCK, I_ZUCK, J_ZUCK}};

use super::encodings::zuck_decode;

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
    data: Vec<Vec<(usize, usize, usize)>>,
    freq_tree: Vec<BinaryHeap<Reverse<Rc<HeapNode>>>>,
    freq_map: Vec<HashMap<usize, usize>>,
    code_len_map: Vec<BTreeMap<usize, BTreeSet<usize>>>,
    // For each context, contains the original data as keys and tuples (code_length, code) as values
    canonical_code_map: Vec<HashMap<usize, (usize, usize)>>,
    /// For each context, contains the number of bits needed to encode the largest datum
    pub code_bits: Vec<usize>,
    /// For each context, contains the number of bits needed to encode the integer representing the number of encoded values
    pub num_values_bits: Vec<usize>, // TODO: use it instead of 16
    /// For each context, contains the number of bits needed to represent the longest encoding
    pub longest_value_bits: Vec<usize>,
    contexts_num: usize,
}

impl HuffmanEncoder {
    /// Creates a new `HuffmanEncoder`.
    /// 
    /// # Arguments
    /// 
    /// * `v` - A bidimensional vector of size `contexts_num` with each inner vector made of tuples containing Zuckerli coding triples.
    /// * `code_bits` - For each context, the number of bits needed to encode the largest datum.
    /// * `num_values_bits` - For each context, the number of bits needed to encode the integer representing the number of encoded values.
    /// * `contexts_num` - The number of contexts.
    pub fn new(v: Vec<Vec<(usize, usize, usize)>>, code_bits: Vec<usize>, num_values_bits: Vec<usize>, contexts_num: usize) -> Self {
        Self { 
            data: v, 
            code_bits, 
            num_values_bits, 
            contexts_num,
            longest_value_bits: vec![0; contexts_num],
            freq_tree: vec![BinaryHeap::default(); contexts_num],
            freq_map: vec![HashMap::default(); contexts_num],
            code_len_map: vec![BTreeMap::default(); contexts_num],
            canonical_code_map: vec![HashMap::default(); contexts_num]
         }
    }

    fn get_frequency(&mut self, context: usize) {
        debug_assert!(context < self.contexts_num);
        
        for &x in self.data[context].iter() {
            self.freq_map[context].entry(x.0).and_modify(|freq| *freq += 1).or_insert(1);
        }
    }

    fn create_huffman_tree(&mut self, context: usize) -> Option<Rc<HeapNode>>{
        debug_assert!(context < self.contexts_num);

        for (&k, &v) in self.freq_map[context].iter() {
            self.freq_tree[context].push(Reverse(Rc::new(HeapNode::new(k, v))));
        }

        let mut root = 
            if self.freq_tree[context].len() == 1 {
                Some(self.freq_tree[context].peek().unwrap().0.clone())
            } else {
                None
            };

        while self.freq_tree[context].len() > 1 {
            let node1 = self.freq_tree[context].pop().unwrap().clone();
            let node2 = self.freq_tree[context].pop().unwrap().clone();

            let mut new_node = HeapNode::new(0, node1.0.data + node2.0.data);
            new_node.set_left_child(Some(node1.0));
            new_node.set_right_child(Some(node2.0));

            let new_node = Rc::new(new_node);
            
            root = Some(new_node.clone());

            self.freq_tree[context].push(Reverse(new_node));
        }

        root
    }

    fn get_code_len(&mut self, root: Option<Rc<HeapNode>>, len: usize, context: usize) {
        debug_assert!(context < self.contexts_num);
        
        if let Some(root) = root {
            if root.left.is_none() && root.right.is_none() {
                self.code_len_map[context].entry(len)
                                    .and_modify(|int_set| { int_set.insert(root.key); })
                                    .or_insert_with(|| { let mut hs = BTreeSet::new(); hs.insert(root.key); hs });
                return;                
            }

            self.get_code_len(root.left.clone(), len + 1, context);
            self.get_code_len(root.right.clone(), len + 1, context);
        }
    }

    fn generate_canonical_code(&mut self, context: usize) {
        debug_assert!(context < self.contexts_num);
        
        let mut prev_len = 0;
        let mut curr_val = 0;
        let mut canonical_len = 0;

        for (&len, int_set) in self.code_len_map[context].iter() {
            canonical_len += len - prev_len;
            curr_val <<= len - prev_len;
            for &i in int_set.iter() {
                self.canonical_code_map[context].insert(i, (if canonical_len == 0 {1} else {canonical_len}, curr_val));
                curr_val += 1;
            }
            prev_len = len;
        }
    }

    /// Writes the canonical Huffman's header relative to a given context onto a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `writer` - The binary stream onto which to write the header.
    /// * `context` - The context to which the header refers to.
    pub fn write_header(&mut self, writer: &mut BinaryWriterBuilder, context: usize) {
        debug_assert!(context < self.contexts_num);
        
        let mut ordered_codes = Vec::with_capacity(self.canonical_code_map[context].len());

        for (&k, &v) in self.canonical_code_map[context].iter() {
            ordered_codes.push((k, v));
        }

        // Order by increasing code length
        ordered_codes.sort_by(|first, second| if first.1.0 == second.1.0 {first.0.cmp(&second.0)} else {first.1.0.cmp(&second.1.0)});

        let max_len = ordered_codes.last().unwrap().1.0;
        let int_num = ordered_codes.len();

        self.longest_value_bits[context] = HuffmanEncoder::get_minimum_amount_bits(max_len);

        writer.push_bits(max_len as u64, self.longest_value_bits[context] as u64); // The first `longest_value_bits` bits will represent the maximum code_length

        let mut search_index = 0;

        for i in 1..max_len + 1 {
            let mut count = 0;
            while search_index < ordered_codes.len() && i == ordered_codes[search_index].1.0 {
                count += 1;
                search_index += 1;
            }
            writer.push_bits(count, max_len as u64 + 1);
        }

        writer.push_bits(int_num as u64, 16); // The total number of unique zuckerli-encoded values // TODO: check correctness for large graphs
        for x in ordered_codes.iter() {
            writer.push_bits(x.0 as u64, self.code_bits[context] as u64);
        }
    }

    pub fn write_headers(&mut self, writer: &mut BinaryWriterBuilder) {
        for ctx in 0..self.contexts_num {
            self.write_header(writer, ctx);
        }
    }

    /// Writes all the encoded integers contained in a context of `self.data` to a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `writer` - The binary stream onto which to write the data.
    /// * `context` - The context to which the body to write refers to.
    pub fn write_body(&self, writer: &mut BinaryWriterBuilder, context: usize) {
        debug_assert!(context < self.contexts_num);
        
        for int in self.data[context].iter() {
            let length = self.canonical_code_map[context].get(&int.0).unwrap().0;
            let code = self.canonical_code_map[context].get(&int.0).unwrap().1;

            writer.push_bits(code as u64, length as u64);
            writer.push_bits(int.2 as u64, int.1 as u64);
        }
    }

    /// Iteratively encodes and writes a value of a context to a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `writer` - The binary stream onto which to write the encoded value.
    /// * `value` - The integer to encode.
    /// * `context` - The context which contains the value.
    #[inline(always)]
    pub fn write(&self, writer: &mut BinaryWriterBuilder, value: usize, context: usize) {
        debug_assert!(context < self.contexts_num);
        
        let (x, zuck_t_len, zuck_t) = zuck_encode(value, K_ZUCK, I_ZUCK, J_ZUCK);

        let item = self.canonical_code_map[context].get(&x);

        debug_assert!(item.is_some(), "The value {} has not been encoded", x);

        let (length, code) = item.unwrap();

        writer.push_bits(*code as u64, *length as u64);
        writer.push_bits(zuck_t as u64, zuck_t_len as u64);
    }
    
    /// Creates the Huffman encoding returning an `HuffmanEncoder`.
    /// 
    /// # Arguments
    /// 
    /// * `data` - The bidimensional vector of contexts containing the data to be encoded.
    pub fn build_huffman(data: &Vec<Vec<usize>>) -> Self {
        let mut transformed_data = Vec::with_capacity(data.len());
        for v in data.iter() {
            transformed_data.push(Vec::with_capacity(v.len()));
        }

        let mut max_data = vec![0; data.len()];
        let mut max_len = vec![0; data.len()];
        let mut max_t = vec![0; data.len()];

        for (i, v) in data.iter().enumerate() {
            for &x in v {
                let zuck_triple = zuck_encode(x, K_ZUCK, I_ZUCK, J_ZUCK);
                transformed_data[i].push(zuck_triple);
                max_data[i] = zuck_triple.0.max(max_data[i]);
                max_len[i] = zuck_triple.1.max(max_len[i]);
                max_t[i] = zuck_triple.2.max(max_t[i]);
            }
        }

        let mut bits_data = Vec::new();
        for md in max_data {
            bits_data.push(HuffmanEncoder::get_minimum_amount_bits(md));
        }
        let bits_values = Vec::new();
        for d in data {
            HuffmanEncoder::get_minimum_amount_bits(d.len());
        }

        let mut huff = HuffmanEncoder::new(transformed_data, bits_data, bits_values, data.len());
        
        for ctx in 0..data.len() {
            huff.get_frequency(ctx);
            let root = huff.create_huffman_tree(ctx);
            huff.get_code_len(root, 0, ctx);
            huff.generate_canonical_code(ctx);
        }

        huff
    }

    /// Builds an `HuffmanEncoder` from a list of hashmaps containing the value and its frequency, one per context.
    /// 
    /// # Arguments
    /// 
    /// * `data` - The bidimensional vector of contexts containing the hashmaps.
    pub fn build_from_frequencies(data: &mut Vec<HashMap<usize, usize>>) -> Self {
        let mut new_data = Vec::with_capacity(data.len());

        for (ctx, hm) in data.into_iter().enumerate() {
            let mut len = 0;
            for val in hm.values() {
                len += val;
            }
            new_data.push(Vec::with_capacity(len));
            for (num, times) in hm {
                for _ in 0..*times {
                    new_data[ctx].push(*num);
                }
            } 
        }

        HuffmanEncoder::build_huffman(&new_data)
    }

    /// Writes the whole content (i.e., each context's header and each context's body) 
    /// of an `HuffmanEncoder` to a binary stream.
    /// 
    /// # Arguments
    /// 
    /// * `encoder` - The `HuffmanEncoder` containing data and its representations.
    /// * `writer` - The binary writer onto which to write.
    pub fn write_huffman_zuck(encoder: &mut HuffmanEncoder, writer: &mut BinaryWriterBuilder) {
        for ctx in 0..encoder.contexts_num {
            encoder.write_header(writer, ctx);
        }
        for ctx in 0..encoder.contexts_num {
            encoder.write_body(writer, ctx);
        }
    }

    #[inline(always)]
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
    canonical_tree_root: Vec<Option<Rc<RefCell<HeapNodeDecoder>>>>,
    code_bits: Vec<usize>,
    to_read: Vec<usize>, // TODO: remove
    longest_value_bits: Vec<usize>,
    num_contexts: usize,
}

impl HuffmanDecoder {
    /// Creates a HuffmanDecoder.
    /// 
    /// # Arguments
    /// 
    /// * `reader` - A shared `RefCell` containing the binary reader to read.
    /// * `code_bits` - For each context, the number of bits needed for the largest datum to decode.
    /// * `longest_value_bits` - For each context, the number of bits needed to write the integer representing the maximum code length.
    /// * `num_contexts` - The number of contexts.
    pub fn new(reader: Rc<RefCell<BinaryReader>>, code_bits: Vec<usize>, longest_value_bits: Vec<usize>, num_contexts: usize) -> Self {
       Self { 
        reader,
        canonical_tree_root: vec![None; num_contexts],
        code_bits,
        to_read: vec![0; num_contexts],
        longest_value_bits,
        num_contexts
        }
    }

    /// Reads the canonical Huffman's header of a given context.
    /// 
    /// # Arguments
    /// 
    /// * `context` - The context to which the header refers to.
    pub fn read_header(&mut self, context: usize) {
        self.to_read[context] = self.reader.as_ref().borrow().is.len();

        let max_len = (*self.reader).borrow_mut().read_int(self.longest_value_bits[context] as u64) as usize;
        let mut length = Vec::with_capacity(max_len);
        for _ in 0..max_len {
            length.push((*self.reader).borrow_mut().read_int(max_len as u64 + 1) as usize);
        }

        let number_of_ints = (*self.reader).borrow_mut().read_int(16) as usize;
        let mut symbols = Vec::with_capacity(number_of_ints);
        for _ in 0..number_of_ints {
            symbols.push((*self.reader).borrow_mut().read_int(self.code_bits[context] as u64) as usize);
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

        self.canonical_tree_root[context] = Some(root);
    }

    pub fn read_headers(&mut self) {
        for ctx in 0..self.num_contexts {
            self.read_header(ctx);
        }
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

    /// Iteratively reads the next Huffman-encoded value belonging to a context.
    /// 
    /// # Arguments
    /// 
    /// * `reader` - The binary reader containing the bits to read.
    /// * `context` - The context to which the value belongs.
    #[inline(always)]
    pub fn read(&mut self, reader: &mut BinaryReader, context: usize) -> usize {
        let mut res = 0;

        let mut curr_node = Some(self.canonical_tree_root[context].as_ref().unwrap().clone());

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
}

#[test]
fn test_huffman_iterative_read() {
    let v = vec![vec![10000, 20000, 65535, 65535, 30000, 1, 1, 20000, 3, 3, 100, 200, 65535, 65535, 1]];

    let mut huff = HuffmanEncoder::build_huffman(&v);

    let mut binary_writer = BinaryWriterBuilder::new();

    HuffmanEncoder::write_huffman_zuck(&mut huff, &mut binary_writer);

    let code_bits = huff.code_bits;
    let longest_value_bits = huff.longest_value_bits;
    let contexts_num = huff.contexts_num;
    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff = HuffmanDecoder::new(reader.clone(), code_bits, longest_value_bits, contexts_num);
    huff.read_header(0);

    let mut result = Vec::new();

    for _ in 0..v[0].len() {
        result.push(huff.read(&mut (*reader).borrow_mut(), 0));
    }

    assert_eq!(result, v[0]);
}

#[test]
fn test_huffman_interleaved() {
    let v1 = vec![vec![10000, 20000, 65535, 65535, 30000]];
    let v2 = vec![vec![1, 1, 20000, 3, 3]];
    let v3 = vec![vec![100, 200, 65535, 65535, 1]];

    let mut huff1 = HuffmanEncoder::build_huffman(&v1);
    let mut huff2 = HuffmanEncoder::build_huffman(&v2);
    let mut huff3 = HuffmanEncoder::build_huffman(&v3);

    let mut binary_writer = BinaryWriterBuilder::new();

    huff1.write_header(&mut binary_writer, 0);
    huff2.write_header(&mut binary_writer, 0);
    huff3.write_header(&mut binary_writer, 0);

    for i in 0..5 {
        huff1.write(&mut binary_writer, v1[0][i], 0);
        huff2.write(&mut binary_writer, v2[0][i], 0);
        huff3.write(&mut binary_writer, v3[0][i], 0);
    }

    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff1 = HuffmanDecoder::new(reader.clone(), huff1.code_bits, huff1.longest_value_bits, huff1.contexts_num);
    let mut huff2 = HuffmanDecoder::new(reader.clone(), huff2.code_bits, huff2.longest_value_bits, huff2.contexts_num);
    let mut huff3 = HuffmanDecoder::new(reader.clone(), huff3.code_bits, huff3.longest_value_bits, huff3.contexts_num);

    huff1.read_header(0);
    huff2.read_header(0);
    huff3.read_header(0);

    let mut result1 = Vec::new();
    let mut result2 = Vec::new();
    let mut result3 = Vec::new();

    for _ in 0..5 {
        result1.push(huff1.read(&mut (*reader).borrow_mut(), 0));
        result2.push(huff2.read(&mut (*reader).borrow_mut(), 0));
        result3.push(huff3.read(&mut (*reader).borrow_mut(), 0));
    }

    assert_eq!(result1, v1[0]);
    assert_eq!(result2, v2[0]);
    assert_eq!(result3, v3[0]);
}

#[test]
fn test_huffman_interleaved_multiple_encodings() {
    let v1 = vec![vec![10000, 20000, 65535, 65535, 30000]];
    let v2 = vec![vec![1, 1, 20000, 3, 3]];
    let v3 = vec![vec![100, 200, 65535, 65535, 1]];

    let mut huff1 = HuffmanEncoder::build_huffman(&v1);
    let mut huff2 = HuffmanEncoder::build_huffman(&v2);
    let mut huff3 = HuffmanEncoder::build_huffman(&v3);

    let mut binary_writer = BinaryWriterBuilder::new();

    huff1.write_header(&mut binary_writer, 0);
    huff2.write_header(&mut binary_writer, 0);
    huff3.write_header(&mut binary_writer, 0);

    for i in 0..5 {
        huff1.write(&mut binary_writer, v1[0][i], 0);
        huff2.write(&mut binary_writer, v2[0][i], 0);
        huff3.write(&mut binary_writer, v3[0][i], 0);
    }

    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff1 = HuffmanDecoder::new(reader.clone(), huff1.code_bits, huff1.longest_value_bits, huff1.contexts_num);
    let mut huff2 = HuffmanDecoder::new(reader.clone(), huff2.code_bits, huff2.longest_value_bits, huff2.contexts_num);
    let mut huff3 = HuffmanDecoder::new(reader.clone(), huff3.code_bits, huff3.longest_value_bits, huff3.contexts_num);

    huff1.read_header(0);
    huff2.read_header(0);
    huff3.read_header(0);

    let mut result1 = Vec::new();
    let mut result2 = Vec::new();
    let mut result3 = Vec::new();

    for _ in 0..5 {
        result1.push(huff1.read(&mut (*reader).borrow_mut(), 0));
        result2.push(huff2.read(&mut (*reader).borrow_mut(), 0));
        result3.push(huff3.read(&mut (*reader).borrow_mut(), 0));
    }

    assert_eq!(result1, v1[0]);
    assert_eq!(result2, v2[0]);
    assert_eq!(result3, v3[0]);
}

#[test]
fn test_huffman_multiple_contexts() {
    let v1 = vec![vec![10000, 20000, 65535, 65535, 30000], vec![50, 60, 90, 50, 50, 1, 60, 90]];
    let v2 = vec![vec![1, 1, 20000, 3, 3]];
    let v3 = vec![vec![100, 200, 65535, 65535, 1], vec![1, 1, 1, 1, 1, 3, 2, 1], vec![10, 12]];

    let mut huff1 = HuffmanEncoder::build_huffman(&v1);
    let mut huff2 = HuffmanEncoder::build_huffman(&v2);
    let mut huff3 = HuffmanEncoder::build_huffman(&v3);

    let mut binary_writer = BinaryWriterBuilder::new();

    huff1.write_headers(&mut binary_writer);
    huff2.write_headers(&mut binary_writer);
    huff3.write_headers(&mut binary_writer);

    for (i, v) in v1.iter().enumerate() {
        for &x in v {
            huff1.write(&mut binary_writer, x, i);
        }
    }
    for (i, v) in v2.iter().enumerate() {
        for &x in v {
            huff2.write(&mut binary_writer, x, i);
        }
    }
    for (i, v) in v3.iter().enumerate() {
        for &x in v {
            huff3.write(&mut binary_writer, x, i);
        }
    }

    let out = binary_writer.build().os;

    let reader = Rc::new(RefCell::new(BinaryReader::new(out.into())));

    let mut huff1 = HuffmanDecoder::new(reader.clone(), huff1.code_bits, huff1.longest_value_bits, huff1.contexts_num);
    let mut huff2 = HuffmanDecoder::new(reader.clone(), huff2.code_bits, huff2.longest_value_bits, huff2.contexts_num);
    let mut huff3 = HuffmanDecoder::new(reader.clone(), huff3.code_bits, huff3.longest_value_bits, huff3.contexts_num);

    huff1.read_headers();
    huff2.read_headers();
    huff3.read_headers();

    let mut result1 = vec![vec![]; huff1.num_contexts];
    let mut result2 = vec![vec![]; huff2.num_contexts];
    let mut result3 = vec![vec![]; huff3.num_contexts];

    for ctx in 0..v1.len() {
        for _ in 0..v1[ctx].len() {
            result1[ctx].push(huff1.read(&mut (*reader).borrow_mut(), ctx));
        }
    }
    for ctx in 0..v2.len() {
        for _ in 0..v2[ctx].len() {
            result2[ctx].push(huff2.read(&mut (*reader).borrow_mut(), ctx));
        }
    }
    for ctx in 0..v3.len() {
        for _ in 0..v3[ctx].len() {
            result3[ctx].push(huff3.read(&mut (*reader).borrow_mut(), ctx));
        }
    }

    assert_eq!(result1, v1);
    assert_eq!(result2, v2);
    assert_eq!(result3, v3);
}

#[test]
fn test_building_from_frequencies() {
    let v1 = vec![vec![10000, 20000, 65535, 65535, 30000], vec![50, 60, 90, 50, 50, 1, 60, 90]];
    let mut freqs: Vec<HashMap<usize, usize>> = vec![HashMap::new(); v1.len()];

    for (ctx, v) in v1.iter().enumerate() {
        for val in v.iter() {
            freqs[ctx].entry(*val).and_modify(|e| *e += 1).or_insert(1);
        }
    }

    let huff1 = HuffmanEncoder::build_huffman(&v1);
    let huff2 = HuffmanEncoder::build_from_frequencies(&mut freqs);

    assert_eq!(huff1.freq_map, huff2.freq_map);
}