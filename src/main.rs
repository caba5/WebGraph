use webgraph_rust::{EncodingType, ImmutableGraph, Properties};
use webgraph_rust::webgraph::{BVGraphBuilder, UnaryCode, GammaCode, DeltaCode, ZetaCode};

use std::fs;
use std::marker::PhantomData;

use clap::Parser;

#[derive(Parser, Debug)]
struct WGArgs { // TODO: automatic compression methods
    /// The size of the window
    window_size: usize,
    /// The maximum reference chain length
    max_ref_count: usize,
    /// The minimum length of the interval
    min_interval_length: usize,
    /// Source filename
    source_name: String,
    /// Destination filename
    dest_name: String,
}

fn main() {
    let args = WGArgs::parse();

    // let props = serde_json::from_str::<Properties>(
    //     fs::read_to_string(format!("{}.properties", args.source_name)).unwrap().as_str()
    // ).unwrap();

    // assert!(props.nodes as u64 <= u64::MAX, "This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", props.nodes);

    // let bvgraph_builder = BVGraphBuilder::<GammaCode, GammaCode, ZetaCode<3>, UnaryCode, GammaCode, GammaCode>{
    //     num_nodes: props.nodes,
    //     num_edges: props.arcs,
    //     loaded_graph: Vec::default(),
    //     loaded_offsets: Vec::default(),
    //     cached_node: None,
    //     cached_outdegree: None,
    //     cached_ptr: None,
    //     max_ref_count: props.max_ref_count,
    //     window_size: props.window_size,
    //     min_interval_len: props.min_interval_len,
    //     zeta_k: props.zeta_k,
    //     outdegree_coding: props.outdegree_coding, 
    //     block_coding: props.block_coding, 
    //     residual_coding: props.residual_coding, 
    //     reference_coding: props.reference_coding, 
    //     block_count_coding: props.block_count_coding, 
    //     offset_coding: props.offset_coding,
    //     _phantom_block_coding: PhantomData,
    //     _phantom_block_count_coding: PhantomData,
    //     _phantom_outdegree_coding: PhantomData,
    //     _phantom_offset_coding: PhantomData,
    //     _phantom_reference_coding: PhantomData,
    //     _phantom_residual_coding: PhantomData,
    // };

    let graph = BVGraphBuilder::<GammaCode, GammaCode, ZetaCode<3>, UnaryCode, GammaCode, GammaCode, usize>::new()
                .load_graph_plain(&args.source_name)
                .load_offsets_plain(&args.source_name)
                .set_window_size(args.window_size)
                .set_ref_count(args.max_ref_count)
                .set_min_interval_len(args.min_interval_length)
                .build();

    graph.store(&args.dest_name).expect("Failed storing the graph");
}