use std::{fs::{File, self}, io::BufReader};

use clap::Parser;
use webgraph_rust::{properties::Properties, ImmutableGraph, bitstreams::BinaryWriter, utils::{encodings::{UnaryCode, ZetaCode, GammaCode, UniversalCode}, EncodingType}, webgraph::bvgraph::BVGraphBuilder};


#[derive(Parser, Debug)]
#[command(about = "Generate the gamma-encoded .offsets file for a graph")]
struct Args {
    /// The basename of the graph file
    source_name: String,
}

fn main() {
    let args = Args::parse();

    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));
    let props = Properties::from(p);
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA) => {},
        _ => panic!("Only the default encoding types sequence (GAMMA, GAMMA, GAMMA, GAMMA, UNARY, GAMMA, ZETA) is supported for generating offsets")
    };

    let loaded_g = BVGraphBuilder::<
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode,
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode, // Dummy values which won't be used
    >::new()
        .set_in_min_interval_len(props.min_interval_len)
        .set_in_max_ref_count(props.max_ref_count)
        .set_in_window_size(props.window_size)
        .set_in_zeta(props.zeta_k)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(&args.source_name)
        // Skip the offsets loading
        .load_outdegrees()
        .build();
    
    let mut offsets_writer = BinaryWriter::new();

    let mut iter = loaded_g.iter();
    let mut n = loaded_g.num_nodes();
    let mut last_offset = 0;

    while n > 0 {
        GammaCode::write_next(&mut offsets_writer, (iter.ibs.read_bits - last_offset) as u64, None);
        last_offset = iter.ibs.read_bits;

        iter.next();
        iter.outdegree();
        iter.successor_array();

        n -= 1;
    }
    GammaCode::write_next(&mut offsets_writer, (iter.ibs.read_bits - last_offset) as u64, None);

    let offsets = offsets_writer.build();

    fs::write(format!("{}.offsets", args.source_name), offsets.os).expect("Failed storing the offsets file");
}