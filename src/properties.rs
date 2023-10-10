use std::collections::HashMap;

use crate::{EncodingType, BitsLen};

#[derive(Debug)]
pub struct Properties {
    pub nodes: usize,
    pub arcs: usize,
    pub window_size: usize,
    pub max_ref_count: usize,
    pub min_interval_len: usize,
    pub zeta_k: Option<u64>,
    pub outdegree_coding: EncodingType,
    pub block_coding: EncodingType,
    pub residual_coding: EncodingType,
    pub reference_coding: EncodingType,
    pub block_count_coding: EncodingType,
    pub offset_coding: EncodingType,
    pub huff_blocks_bits: BitsLen,
    pub huff_residuals_bits: BitsLen,
    pub huff_intervals_bits: BitsLen,
}

impl Default for Properties {
    fn default() -> Self {
        Self { 
            nodes: 0, 
            arcs: 0, 
            window_size: 7, 
            max_ref_count: 3, 
            min_interval_len: 4, 
            zeta_k: Some(3), 
            outdegree_coding: EncodingType::GAMMA, 
            block_coding: EncodingType::GAMMA, 
            residual_coding: EncodingType::ZETA, 
            reference_coding: EncodingType::UNARY, 
            block_count_coding: EncodingType::GAMMA, 
            offset_coding: EncodingType::GAMMA,
            huff_blocks_bits: BitsLen::default(),
            huff_residuals_bits: BitsLen::default(),
            huff_intervals_bits: BitsLen::default()
        }
    }
}

impl From<HashMap<String, String>> for Properties {
    fn from(value: HashMap<String, String>) -> Self {
        let mut props = Properties {
            nodes: value.get("nodes").expect("Failed in reading nodes from properties").parse().unwrap(),
            arcs: value.get("arcs").expect("Failed in reading arcs from properties").parse().unwrap(),
            window_size: value.get("windowsize").expect("Failed in reading window size from properties").parse().unwrap(),
            max_ref_count: value.get("maxrefcount").expect("Failed in reading maximum reference count from properties").parse().unwrap(),
            min_interval_len: value.get("minintervallength").expect("Failed in reading minimum interval length from properties").parse().unwrap(),
            ..Default::default()
        };

        if let Some(zeta_k) = value.get("zetak") {
            props.zeta_k = Some(zeta_k.parse().unwrap())
        }
        if let Some(huff_blocks_bits) = value.get("huff_blocks_bits") {
            if !huff_blocks_bits.is_empty() {
                let s: Vec<_> = huff_blocks_bits.split(',').collect();

                props.huff_blocks_bits = 
                    BitsLen { 
                        code_bits: s[0].trim().parse().unwrap(),
                        longest_value_bits: s[1].trim().parse().unwrap()
                    };
            }
        }
        if let Some(huff_residuals_bits) = value.get("huff_residuals_bits") {
            if !huff_residuals_bits.is_empty() {
                let s: Vec<_> = huff_residuals_bits.split(',').collect();

                props.huff_residuals_bits = 
                    BitsLen { 
                        code_bits: s[0].trim().parse().unwrap(),
                        longest_value_bits: s[1].trim().parse().unwrap()
                    };
            }
        }
        if let Some(huff_intervals_bits) = value.get("huff_intervals_bits") {
            if !huff_intervals_bits.is_empty() {
                let s: Vec<_> = huff_intervals_bits.split(',').collect();

                props.huff_intervals_bits = 
                    BitsLen { 
                        code_bits: s[0].trim().parse().unwrap(),
                        longest_value_bits: s[1].trim().parse().unwrap()
                    };
            }
        }

        if let Some(compression_flags) = value.get("compressionflags") {
            if !compression_flags.is_empty() {
                for flag in compression_flags.split('|') {
                    let s: Vec<_> = flag.split('_').collect();

                    // if s.last().unwrap().trim().to_uppercase() == "HUFFMAN" {
                    //     continue;
                    // }

                    match s[0].trim().to_uppercase().as_str() {
                        "OUTDEGREES" => props.outdegree_coding = EncodingType::from(s[1]),
                        "REFERENCES" => props.reference_coding = EncodingType::from(s[1]),
                        "BLOCKS" => props.block_coding = EncodingType::from(s[1]),
                        "RESIDUALS" => props.residual_coding = EncodingType::from(s[1]),
                        "OFFSETS" => props.offset_coding = EncodingType::from(s[1]),
                        "BLOCK" => props.block_count_coding = EncodingType::from(s[2]),
                        _ => panic!("Unkown compression flag {}", s[0]),
                    };
                }
            }
        }

        props
    }
}

impl From<Properties> for String {
    fn from(val: Properties) -> Self {
        let mut s = String::new();

        s.push_str("#BVGraph properties\n");
        s.push_str("version=0\n");
        s.push_str("graphclass=it.unimi.dsi.webgraph.BVGraph\n");
        s.push_str(&format!("nodes={}\n", val.nodes));
        s.push_str(&format!("arcs={}\n", val.arcs));
        s.push_str(&format!("minintervallength={}\n", val.min_interval_len));
        s.push_str(&format!("maxrefcount={}\n", val.max_ref_count));
        s.push_str(&format!("windowsize={}\n", val.window_size));
        s.push_str(&format!("zetak={}\n", if val.zeta_k.is_some() {val.zeta_k.unwrap()} else {3}));
        s.push_str("compressionflags=");

        let mut cflags = false;

        if val.outdegree_coding != EncodingType::GAMMA {
            s.push_str(&format!("OUTDEGREES_{} |", val.outdegree_coding));
            cflags = true;
        }
        if val.reference_coding != EncodingType::UNARY {
            s.push_str(&format!("REFERENCES_{} |", val.reference_coding));
            cflags = true;
        }
        if val.block_coding != EncodingType::GAMMA {
            s.push_str(&format!("BLOCKS_{} |", val.block_coding));
            cflags = true;
        }
        if val.block_count_coding != EncodingType::GAMMA {
            s.push_str(&format!("BLOCK_COUNT_{} |", val.block_count_coding));
            cflags = true;
        }
        if val.residual_coding != EncodingType::ZETA {
            s.push_str(&format!("RESIDUALS_{} |", val.residual_coding));
            cflags = true;
        }
        if val.offset_coding != EncodingType::GAMMA {
            s.push_str(&format!("OFFSETS_{} |", val.offset_coding));
            cflags = true;
        }

        if cflags {
            s.pop();
            s.pop();
        }

        s.push('\n');

        s.push_str(&format!("huff_blocks_bits={}\n", val.huff_blocks_bits));
        s.push_str(&format!("huff_residuals_bits={}\n", val.huff_residuals_bits));
        s.push_str(&format!("huff_intervals_bits={}\n", val.huff_intervals_bits));

        s
    }
}