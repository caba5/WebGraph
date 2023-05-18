mod tests;

use std::fs;

use serde::{Serialize, Deserialize};

pub trait ImmutableGraph {
    fn num_nodes(&self) -> u32;
    fn num_arcs(&self) -> u32;
    fn successors(&self) -> iter;
    // fn outdegree(&self, x: &i32) -> usize;
    // fn node_iterator(&self) -> iter;
    // fn outdegrees(&self) -> iter;
    // TODO: how to use fn store here and solve the problem of serializing 'self'
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub enum EncodingType {
    GAMMA,
    DELTA,
    ZETA,
    NIBBLE,
    GOLOMB,
    UNARY,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph {
    n: u32,
    m: u32,
    pub graph: Vec<u32>,    // Unique list of successors
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

    fn successors(&self) -> LazyIntIterator {
        
    }
}

impl BVGraph {
    fn new() -> BVGraph {
        BVGraph { 
            n: 8, 
            m: 11, 
            graph: Vec::new(), 
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

    fn load(name: &str) -> BVGraph {
        let graph = fs::read(name);
        if let Ok(data) = graph {
            return bincode::deserialize::<BVGraph>(&data).unwrap(); // TODO: how to put this function into the trait?
        } else {
            todo!()
        }        
    }

    fn store(&self, name: &str) {
        let serialized = bincode::serialize(self).unwrap();
        fs::write(name, serialized).unwrap();
    }

    
}

// impl ToString for BVGraph {

// }

impl Iterator for BVGraph { // TODO: is it possible to put ImmutableGraph instead of BVGraph?
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()    
    }

    fn skip(self, n: usize) -> std::iter::Skip<Self>
        where
            Self: Sized, {
        todo!()
    }
}