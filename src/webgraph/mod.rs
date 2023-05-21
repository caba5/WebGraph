mod tests;

use std::{fs::{self, File}, cmp::min, io::{BufReader, BufRead, Read}};

use serde::{Serialize, Deserialize};

pub trait ImmutableGraph {
    fn num_nodes(&self) -> u32;
    fn num_arcs(&self) -> u32;
    fn outdegree(&self, x: u32) -> Result<u32, String>;  // TODO: better error
    fn successors(&self, x: u32) -> Result<Box<dyn Iterator<Item = u32>>, &str>; // TODO: Is it right to box?
    // fn node_iterator(&self) -> iter;
    // fn outdegrees(&self) -> iter;
    // TODO: how to use fn store here and solve the problem of serializing 'self'
}

#[derive(Serialize, Deserialize)]
struct Properties {
    nodes: u32,
    tot_links: u32,
    arcs: u32,
    window_size: u32,
    max_ref_count: u32,
    min_interval_len: u32,
    zetak: u32,
    comp: EncodingType,
    avg_ref: f32,
    avg_dist: f32,
    copied_arcs: u32,
    intervalized_arcs: u32,
    residual_arcs: u32,
    bits_per_link: f32,
    comp_ratio: f32,
    bits_per_node: f32,
    avg_bits_for_outdeg: f32,
    avg_bits_for_refs: f32,
    avg_bits_for_blocks: f32,
    avg_bits_for_residuals: f32,
    avg_bits_for_intervals: f32,
    bits_for_outdeg: u32,
    bits_for_refs: u32,
    bits_for_blocks: u32,
    bits_for_residuals: u32,
    bits_for_intervals: u32,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub enum EncodingType {  
    GAMMA,
    DELTA,
    ZETA,
    NIBBLE,
    GOLOMB,
    SKEWEDGOLOMB, // Taken from CompressionFlags.java
    UNARY,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph {
    n: u32, // TODO: private with getter or public?
    m: u32,
    pub graph_memory: Vec<u8>,    // Unique list of successors
    pub offsets: Vec<u32>,  // Each offset at position i indicates where does node i start in 'graph'. TODO: does it have to be stored separately as in the original code?
    pub cached_node: u32,
    pub cached_outdegree: u32,
    pub cached_ptr: usize,
    pub max_ref_count: u32,
    pub window_size: u32,
    pub min_interval_len: u32,
    pub zeta_k: u32,
    pub outdegree_coding: EncodingType,
    pub block_coding: EncodingType,
    pub residual_coding: EncodingType,
    pub reference_coding: EncodingType,
    pub block_count_coding: EncodingType,
    pub offset_coding: EncodingType,
}

impl ImmutableGraph for BVGraph {
    fn num_nodes(&self) -> u32 {
        self.n
    }

    fn num_arcs(&self) -> u32 {
        self.m
    }

    fn outdegree(&self, x: u32) -> Result<u32, String> {
        if x == self.cached_node {
            return Ok(self.cached_node);
        }
        if x < 0 || x >= self.n {
            return Err(format!("Node index out of range {}", x));
        }

        outdegree_reader = BufReader::new(self.graph_memory); // TODO

        // outdegreeIbs.position(offsets.getLong(cachedNode = x));
        // cachedOutdegree = readOutdegree(outdegreeIbs);
        // cachedPointer = outdegreeIbs.position();
        // return cachedOutdegree;


    }

    fn successors(&self, x: u32) -> Result<Box<dyn Iterator<Item = u32>>, &str> {
        todo!()
    }
}

impl BVGraph {
    fn new() -> BVGraph {
        BVGraph { 
            n: 8, 
            m: 11, 
            graph_memory: Vec::new(), 
            offsets: Vec::new(), 
            cached_node: 0, 
            cached_outdegree: 0, 
            cached_ptr: 0, 
            max_ref_count: 0, 
            window_size: 0, 
            min_interval_len: 9999, 
            zeta_k: 0, 
            outdegree_coding: EncodingType::GAMMA, 
            block_coding: EncodingType::GAMMA, 
            residual_coding: EncodingType::ZETA, 
            reference_coding: EncodingType::UNARY, 
            block_count_coding: EncodingType::GAMMA, 
            offset_coding: EncodingType::GAMMA 
        }
    }

    pub fn load(name: &str) -> BVGraph {
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

        BVGraph::new()
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
        fs::write(format!("{}.graph", &name), bincode::serialize(self).unwrap()).unwrap(); // TODO
    }


    pub fn successors(&self, idx: u32, reader: &BufReader<Vec<u8>>, window: &Option<Vec<Vec<u32>>>, outd: &Option<Vec<u32>>) {

    }
    
}

#[derive(Clone)]
pub struct ImmutableGraphNodeIterator<'a> {
    graph: &'a (dyn ImmutableGraph + 'a),
    from: u32,
    to: u32,
    curr: u32,
}

impl<'a> ImmutableGraphNodeIterator<'a> {
    pub fn new(graph: &dyn ImmutableGraph, from: u32, to: u32) -> ImmutableGraphNodeIterator {
        ImmutableGraphNodeIterator { 
            graph, 
            from, 
            to: min(graph.num_nodes(), to), 
            curr: from - 1 
        }
    }

    fn has_next(&mut self) -> bool {
        self.curr < self.to - 1
    }

    // TODO: Is box necessary?
    // TODO: Return a proper error
    fn successors(&mut self) -> Result<Box<dyn Iterator<Item = u32> + 'a>, &str> { 
        if self.curr == self.from - 1 {
            return Err("Illegal state exception");
        }
        
        Ok(self.graph.successors(self.curr))
    }

    // TODO: Return a proper error
    fn outdegree(&mut self) -> Result<u32, &str> {
        if self.curr == self.from - 1 {
            return Err("Illegal state exception");
        }

        Ok(self.graph.outdegree(self.curr).ok().unwrap()) // TODO: fix double Result
    }
}

impl<'a> Iterator for ImmutableGraphNodeIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() { 
            return None;
        } 

        self.curr += 1;
        Some(self.curr)
    }
}

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