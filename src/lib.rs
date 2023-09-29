use std::{fmt::Display, str::FromStr};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub mod webgraph;
pub mod bitstreams;
pub mod uncompressed_graph;
pub mod plain_webgraph;
pub mod properties;
pub mod utils;

pub trait ImmutableGraph {
    type NodeT;

    fn num_nodes(&self) -> usize;
    fn num_arcs(&self) -> usize;
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize>; 
    // fn successors(&self, x: Self::NodeT) -> Box<dyn Iterator<Item = &u32>>; // TODO: remove dyn, use a specific iterator type
    // fn node_iterator(&self) -> iter;
    fn store(&mut self, filename: &str) -> std::io::Result<()>;
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Debug, ValueEnum)]
pub enum EncodingType {
    GAMMA,
    DELTA,
    ZETA,
    UNARY,
    HUFFMAN,
}

impl From<&str> for EncodingType {
    fn from(value: &str) -> Self {
        let sanitized = value.trim().to_uppercase();

        match sanitized.as_str() {
            "GAMMA" => EncodingType::GAMMA,
            "DELTA" => EncodingType::DELTA,
            "ZETA" => EncodingType::ZETA,
            "UNARY" => EncodingType::UNARY,
            "HUFFMAN" => EncodingType::HUFFMAN,
            _ => panic!("Encoding type {} is not supported", sanitized)
        }
    }
}

impl Display for EncodingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EncodingType::GAMMA => "gamma",
            EncodingType::DELTA => "delta",
            EncodingType::ZETA => "zeta",
            EncodingType::UNARY => "unary",
            EncodingType::HUFFMAN => "huffman",
        })
    }
}

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct BitsLen {
    code_bits: usize,
    num_values_bits: usize,
}

impl BitsLen {
    fn new(code_bits: usize, num_values_bits: usize) -> Self {
        Self { code_bits, num_values_bits }
    }
}

impl Display for BitsLen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("{}, {}", self.code_bits, self.num_values_bits).as_str())
    }
}

/// Maps integers bijectively into natural numbers.
/// 
/// This method will map a negative integer `x` to `-2x - 1` and a 
/// nonnegative integer `x` to `2x`.
#[inline(always)]
pub fn int2nat(x: i64) -> u64 {
    ((x << 1) ^ (x >> (i64::BITS - 1))) as u64
}

/// Maps natural numbers bijectively into integers.
/// 
/// This method computes the inverse of [`int2nat()`].
#[inline(always)]
pub fn nat2int(x: u64) -> i64 {
    (x as i64 >> 1) ^ !((x as i64 & 1) - 1)
}