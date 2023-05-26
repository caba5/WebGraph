mod tests;

use std::{fs::{self, File}, cmp::min, io::{BufReader, BufRead, Read}};

use serde::{Serialize, Deserialize};

use crate::{ImmutableGraph, EncodingType, Properties};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph<T> where T: num_traits::PrimInt { // TODO: does it need Into<usize>?
    n: usize,
    m: usize,
    graph_memory: Vec<u8>,    // Unique list of bits representing the whole graph's adj. lists
    offsets: Vec<u64>,  // Each offset at position i indicates where does node i start in 'graph'. TODO: it is converted from an EliasFanoLongMonotoneList
    cached_node: T,
    cached_outdegree: usize,
    cached_ptr: usize,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: usize,
    outdegree_coding: EncodingType,
    block_coding: EncodingType,
    residual_coding: EncodingType,
    reference_coding: EncodingType,
    block_count_coding: EncodingType,
    offset_coding: EncodingType,
}

impl<T> ImmutableGraph for BVGraph<T> where T: num_traits::PrimInt + PartialOrd<usize> {
    type NodeT = T;

    fn num_nodes(&self) -> usize {
        self.n
    }

    fn num_arcs(&self) -> usize {
        self.m
    }

    fn outdegree(&self, x: Self::NodeT) -> Option<usize> {
        if x == self.cached_node {
            return Some(self.cached_outdegree);
        }
        
        if x < 0 || x >= self.n {
            return None;
        }

        self.cached_node = x;
        let outdegree_iter = self.graph_memory.iter();  // TODO: need to iterate on bits
        // TODO: outdegree_iter should work as https://github.com/vigna/dsiutils/blob/master/src/it/unimi/dsi/io/InputBitStream.java#L774 since it is delta-coded
        // outdegree_iter.nth(self.offsets[self.cached_node as usize] as usize); // TODO: WebGraph uses 'getLong' from offsets, which is a 'LongBigList'
        // self.cached_outdegree = self.read_outdegree(&outdegree_iter).unwrap();  // TODO: manage error
        // self.cached_ptr = outdegree_iter.position(); // TODO: this should return the current position as in https://github.com/vigna/dsiutils/blob/master/src/it/unimi/dsi/io/InputBitStream.java#L823
        Some(self.cached_outdegree)
    }

    fn successors(&self, x: Self::NodeT) -> Result<Box<dyn Iterator<Item = &u32>>, &str> {
        if x < 0 || x > self.n {
            return  Err("Node index out of range");
        }

        let graph_iter = self.graph_memory.iter();  // TODO: need to iterate on bits
        todo!()
        // return self.successors_internal(x, &graph_iter, Option::None, Option::None);
    }
}

impl<T> BVGraph<T> where T: num_traits::PrimInt {
    // Mockup object
    // fn new() -> BVGraph {
    //     Self { 
    //         n: 8, 
    //         m: 11, 
    //         graph_memory: Vec::new(), 
    //         offsets: Vec::new(), 
    //         cached_node: 0, 
    //         cached_outdegree: 0, 
    //         cached_ptr: 0, 
    //         max_ref_count: 0, 
    //         window_size: 0, 
    //         min_interval_len: 9999, 
    //         zeta_k: 0, 
    //         outdegree_coding: EncodingType::GAMMA, 
    //         block_coding: EncodingType::GAMMA, 
    //         residual_coding: EncodingType::ZETA, 
    //         reference_coding: EncodingType::UNARY, 
    //         block_count_coding: EncodingType::GAMMA, 
    //         offset_coding: EncodingType::GAMMA 
    //     }
    // }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_outdegree<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        match self.outdegree_coding {
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => Err("The encoding is not supported")
        }
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_block<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        match self.outdegree_coding {
            EncodingType::UNARY => todo!(), // TODO: implement outdegree_iter.read_unary()
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => Err("The encoding is not supported")
        }
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_block_count<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        match self.outdegree_coding {
            EncodingType::UNARY => todo!(), // TODO: implement outdegree_iter.read_unary()
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => Err("The encoding is not supported")
        }
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_reference<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        let refer: u32 = match self.reference_coding {
            EncodingType::UNARY => todo!(), // TODO: implement outdegree_iter.read_unary()
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => return Err("The encoding is not supported")
        };

        // if refer > self.window_size {
        //     return Err("The required reference is incompatible with the windows size");
        // }

        Ok(refer)
    }

    // TODO: may merge the internal with the external
    // fn successors_internal<'a>(
    //     &self, x: u32, 
    //     graph_iter: &impl Iterator<Item = &'a u8>, 
    //     window: Option<Vec<Vec<u32>>>, 
    //     outd: Option<Vec<u32>>
    // ) -> Result<Box<dyn Iterator<Item = &u32>>, &str> {
    //     let d;
    //     let refer;
    //     let refer_index;
    //     let block_count;
    //     let mut block;
    //     let extra_count;
    //     let cyclic_buffer_size = self.window_size + 1;
        
    //     match window {
    //         Option::None => {
    //             d = self.outdegree(x).unwrap();
    //             graph_iter.nth(self.cached_ptr);
    //         },
    //         Some(_) => {
    //             let outdegs = self.read_outdegree(&graph_iter).unwrap(); // TODO: pass the bit iterator
    //             outd.unwrap()[(x % cyclic_buffer_size) as usize] = outdegs;
    //             d = outdegs;
    //         }
    //     }

    //     if d == 0 {
    //         return Ok(Box::new([].iter())); // TODO: create an enum which represents an empty iterator
    //     }

    //     refer = if self.window_size > 0 {self.read_reference(&graph_iter).unwrap() as i32} else {-1};   // TODO: pass the bit iterator
        
    //     refer_index = ((x + cyclic_buffer_size) as i32 - refer) % cyclic_buffer_size as i32;

    //     if refer > 0 {
    //         block_count = self.read_block_count(&graph_iter).unwrap(); // TODO: pass the bit iterator

    //         if block_count != 0 {
    //             block = Vec::with_capacity(usize::try_from(block_count).unwrap());
    //         }

    //         let (mut copied, mut total) = (0, 0);
    //         for i in 0..block_count as usize {
    //             block[i] = self.read_block(&graph_iter).unwrap() + if i == 0 {0} else {1}; // TODO: pass the bit iterator
    //             total += block[i];
    //             if (i & 1) == 0 {
    //                 copied += block[i];
    //             }
    //         }

    //         if (block_count & 1) == 0 {
    //             copied += if window != Option::None {outd.unwrap()[refer_index as usize]} else {self.outdegree(x - refer as u32).unwrap()} - total;
    //         }

    //         extra_count = d - copied;
    //     } else {
    //         extra_count = d;
    //     }

    //     let interval_count;
    //     let left;
    //     let len;

    //     if extra_count > 0 {
    //         interval_count = todo!();  // TODO: graph_iter.read_gamma()
    //         if self.min_interval_len != 0 && interval_count != 0 {
    //             let prev = 0;
    //             left = Vec::with_capacity(interval_count);
    //             len = Vec::with_capacity(interval_count);

    //             prev = x + todo!();  // TODO: graph_iter.read_long_gamma()
    //             left[0] = prev;
    //             len[0] = self.min_interval_len + todo!(); // TODO: graph_iter.read_gamma()

    //             prev += len[0];
    //             extra_count -= len[0];

    //             for i in 1..interval_count {
    //                 prev = prev + 1 + todo!();  // TODO: graph_iter.read_gamma()
    //                 left[i] = prev;
    //                 len[i] = self.min_interval_len + todo!(); // TODO: graph_iter.read_gamma()
    //                 prev += len[i];
    //                 extra_count -= len[i];
    //             }
    //         }
    //     }

    //     let residual_count = extra_count;

    //     let residual_iter = if residual_count == 0 {Option::None} else {todo!()}; // TODO: create ResidualIntIterator

    //     let extra_iter = if interval_count == 0 {residual_iter} else {if residual_count == 0 {todo!()} else {todo!()}};  // TODO: create IntIntervalSequenceIterator & MergedIntIterator

    //     let block_iter = if refer <= 0 {Option::None} else {todo!()};  // TODO: create MaskedIntIterator

    //     if refer <= 0 {
    //         return Ok(Box::new(extra_iter));
    //     }
        
    //     if extra_iter == Option::None {block_iter} else {todo!()} // TODO: MergedIntIterator
    // }

    pub fn load(name: &str) -> BVGraph<T> {
        // Read properties file. TODO: how to represent it (JSON)?
        let graph_props = fs::read(format!("{}.properties", &name)).unwrap();
        let graph_props: Properties = serde_json::from_slice(&graph_props.as_slice()).unwrap();
        
        // Create an input stream to the graph file
        let graph_f = File::open(format!("{}.graph", &name)).unwrap();

        let file_size = graph_f.metadata().unwrap().len();

        let mut graph_memory: Vec<u8>;
        if file_size <= u64::MAX {
            graph_memory = Vec::with_capacity(usize::try_from(file_size).unwrap());
            let mut file_reader = BufReader::new(graph_f);
            file_reader.read_to_end(&mut graph_memory).unwrap(); // Should read the whole graph into memory. Potentially inefficient
        } else {
            todo!() // WebGraph uses "FastMultiByteArrayInputStream" that multiplexes the input content across multiple arrays
        }

        // Create an input stream to the offsets
        let offsets_f = File::open(format!("{}.offsets", &name)).unwrap();
        let offsets_reader = BufReader::new(offsets_f);
        // TODO: It wraps the input stream into an 'EliasFanoMonotoneLongBigList" https://github.com/vigna/Sux4J/blob/master/src/it/unimi/dsi/sux4j/util/EliasFanoMonotoneLongBigList.java#L313

        // TODO: Need to create an 'InputBitStream' around graph_memory

        todo!()
    }


    pub fn store(&self, name: &str) {

        // ...compression perfomed by threads...

        // Writes graph properties in JSON (JSON?)
        let props = Properties{ 
            nodes: self.n, 
            arcs: self.m, 
            window_size: self.window_size, 
            max_ref_count: self.max_ref_count, 
            min_interval_len: self.min_interval_len, 
            zetak: self.zeta_k, 
            tot_links: 0,
            comp: EncodingType::ZETA, 
            avg_ref: 0., 
            avg_dist: 0., 
            copied_arcs: 0, 
            intervalized_arcs: 0, 
            residual_arcs: 0, 
            bits_per_link: 0., 
            comp_ratio: 0., 
            bits_per_node: 0., 
            avg_bits_for_outdeg: 0., 
            avg_bits_for_refs: 0.,
            avg_bits_for_blocks: 0., 
            avg_bits_for_residuals: 0., 
            avg_bits_for_intervals: 0.,
            bits_for_outdeg: 0, 
            bits_for_refs: 0, 
            bits_for_blocks: 0, 
            bits_for_residuals: 0, 
            bits_for_intervals: 0
        };

        let json_props = serde_json::to_string(&props).unwrap(); // TODO: should it return a Result?
        fs::write(format!("{}.properties", &name), json_props).unwrap();
        // fs::write(format!("{}.graph", &name), bincode::serialize(self).unwrap()).unwrap(); // TODO
    }
    
}

// #[derive(Clone)]
// pub struct ImmutableGraphNodeIterator<'a> {
//     graph: &'a (dyn ImmutableGraph + 'a),
//     from: u32,
//     to: u32,
//     curr: u32,
// }

// impl<'a> ImmutableGraphNodeIterator<'a> {
//     pub fn new(graph: &dyn ImmutableGraph, from: u32, to: u32) -> ImmutableGraphNodeIterator {
//         ImmutableGraphNodeIterator { 
//             graph, 
//             from, 
//             to: min(graph.num_nodes(), to), 
//             curr: from - 1 
//         }
//     }

//     fn has_next(&mut self) -> bool {
//         self.curr < self.to - 1
//     }

//     // TODO: Is box necessary?
//     // TODO: Return a proper error
//     fn successors(&mut self) -> Result<Box<dyn Iterator<Item = u32>>, &str> { 
//         if self.curr == self.from - 1 {
//             return Err("Illegal state exception");
//         }
        
//         Ok(self.graph.successors(self.curr))
//     }

//     // TODO: Return a proper error
//     fn outdegree(&mut self) -> Result<u32, &str> {
//         if self.curr == self.from - 1 {
//             return Err("Illegal state exception");
//         }

//         Ok(self.graph.outdegree(self.curr).ok().unwrap()) // TODO: fix double Result
//     }
// }

// impl<'a> Iterator for ImmutableGraphNodeIterator<'a> {
//     type Item = u32;

//     fn next(&mut self) -> Option<Self::Item> {
//         if !self.has_next() { 
//             return None;
//         } 

//         self.curr += 1;
//         Some(self.curr)
//     }
// }

// impl ToString for BVGraph {

// }

// pub struct BVGraphNodeIterator {
//     n: u32,
//     bit_stream: Vec<u8>, // Is it really a vector of bytes?
//     cyclic_buffer_size: u32,
//     window: Vec<Vec<u32>>,
//     outd: Vec<u32>,
//     from: u32,
//     curr: u32,
//     has_next_limit: u32,
// }

// impl BVGraphNodeIterator {
//     fn new(
//         n: u32, 
//         bit_stream: &Vec<u8>, 
//         window_size: u32, 
//         initial_successor_list_len: u32, 
//         from: u32, 
//         upper_bound: u32, 
//         stream_pos: usize, 
//         in_window: Option<&Vec<Vec<u32>>>, 
//         in_outd: Vec<u32>
//     ) -> BVGraphNodeIterator 
//     {
//         assert!(from > 0 && from < n);
//         let cyclic_buffer_size = window_size + 1;
//         let mut window: Vec<Vec<u32>> = vec![vec![0; initial_successor_list_len as usize]; cyclic_buffer_size as usize];
//         let mut outd = vec![0; cyclic_buffer_size as usize];

//         let ibs = &bit_stream[stream_pos..]; // TODO: mutate the bit_stream directly
//         match in_window {
//             Some(in_window) => {
//                 for i in 0..in_window.len() {
//                     window[i] = in_window[i].clone();  //
//                     outd = in_outd.clone(); // TODO: is clone the only way?
//                 }
//             },
//             None if from != 0 => {
//                 let pos;
//                 for i in 0..min(from + 1, cyclic_buffer_size) {
//                     pos = (from - i + cyclic_buffer_size) % cyclic_buffer_size;
//                     todo!()
//                 }
//             }
//             None => ()
//         }

//         todo!()
//     }
// }