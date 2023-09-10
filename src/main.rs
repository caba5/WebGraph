use webgraph_rust::bitstreams::BinaryReader;
use webgraph_rust::{EncodingType, ImmutableGraph, Properties};
use webgraph_rust::webgraph::{BVGraphBuilder, UnaryCode, GammaCode, DeltaCode, ZetaCode, UniversalCode, BVGraph};

use core::panic;
use std::fmt::Display;
use std::fs;
use std::time::Instant;

use clap::Parser;

use rand::Rng;

#[derive(Parser, Debug)]
struct WGArgs {                                             // TODO: add compression options
    /// The size of the window
    #[arg(short = 'w', long = "window-size")]
    window_size: Option<usize>,
    /// The maximum reference chain length
    #[arg(short = 'r', long = "max-ref-count")]
    max_ref_count: Option<usize>,
    /// The minimum length of the interval
    #[arg(short = 'i', long = "min-interval-len")]
    min_interval_length: Option<usize>,
    /// Source filename
    source_name: String,
    /// Destination filename
    dest_name: Option<String>,
    /// Check the compression correctness
    #[arg(short, long = "check", default_value_t = false)]
    check: bool,
    /// Performance test
    #[arg(short, long = "perf", default_value_t = false)]
    perf_test: bool,
}

const N_QUERIES: usize = 1000000;

fn gen_queries(n_queries: usize, range_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..n_queries)
        .map(|_| rng.gen_range(0..range_size))
        .collect()
}

fn decompression_perf_test<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
>(bvgraph: &mut BVGraph<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding>)
{
    let n = bvgraph.num_nodes();

    let queries = gen_queries(N_QUERIES, n - 1);

    let total = Instant::now();
    for &query in queries.iter() {
        bvgraph.successors(query);
    }
    let avg_query = (total.elapsed().as_nanos() as f64) / N_QUERIES as f64;

    println!("time per query: {}ns", avg_query);
}

fn create_graph<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
>(props: &Properties, in_name: &str, out_name: Option<String>, perf_test: bool, check: bool) {
    let mut bvgraph = BVGraphBuilder::<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding>::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(in_name)
        .load_offsets(in_name)
        .load_outdegrees()
        .build();

    if perf_test {
        decompression_perf_test(&mut bvgraph);
    } else if let Some(out_name) = out_name{
        bvgraph.store(out_name.as_str()).expect("Failed storing the graph");

        if check {
            let compressed_graph = BVGraphBuilder::<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding>::new()
                .set_min_interval_len(props.min_interval_len)
                .set_max_ref_count(props.max_ref_count)
                .set_window_size(props.window_size)
                .set_zeta(props.zeta_k)
                .set_num_nodes(props.nodes)
                .set_num_edges(props.arcs)
                .load_graph(out_name.as_str())
                .load_offsets(out_name.as_str())
                .load_outdegrees()
                .build();
            
            assert_eq!(bvgraph.graph_memory, compressed_graph.graph_memory);
            assert_eq!(bvgraph.offsets, compressed_graph.offsets);

            println!("Check passed");
        }
    } else {
        panic!("Neither an outname nor a performance test flag were provided");
    }
}

fn main() {
    let args = WGArgs::parse();

    if args.perf_test && args.check {
        panic!("Both performance test and compression test flags were provided");
    }

    if args.check && args.dest_name.is_none() {
        panic!("Cannot execute a correctness check without compressing to file");
    }

    if args.perf_test && args.dest_name.is_some() {
        println!("Warning: both destination name and performance test flag provided, nothing will be written");
    }

    if args.source_name.ends_with(".plain") { // TODO: wrong, need a way to check if plain
        todo!()
    }

    let mut props = serde_json::from_str::<Properties>(
        fs::read_to_string(
            format!("{}.properties", args.source_name))
            .unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name)
        ).as_str()
    ).unwrap_or_else(|_| panic!("Failed deserializing {}.properties", args.source_name));

    assert!(props.nodes as u64 <= u64::MAX, "This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", props.nodes);

    if let Some(window_size) = args.window_size {
        props.window_size = window_size;
    }
    if let Some(max_ref_count) = args.max_ref_count {
        props.max_ref_count = max_ref_count;
    }
    if let Some(min_interval_length) = args.min_interval_length {
        props.min_interval_len = min_interval_length;
    }

    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode>(&props, &args.source_name, args.dest_name, args.perf_test, args.check),
        _ => panic!("Unexpected encoding types")
        }

    // let graph = BVGraphBuilder::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, usize>::new()
    //             .load_graph_plain(&args.source_name)
    //             .load_offsets_plain(&args.source_name)
    //             .set_window_size(args.window_size)
    //             .set_max_ref_count(args.max_ref_count)
    //             .set_min_interval_len(args.min_interval_length)
    //             .build();
}