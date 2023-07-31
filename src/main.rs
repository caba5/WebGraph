use webgraph_rust::bitstreams::BinaryReader;
use webgraph_rust::{EncodingType, ImmutableGraph, Properties};
use webgraph_rust::webgraph::{BVGraphBuilder, UnaryCode, GammaCode, DeltaCode, ZetaCode, UniversalCode, BVGraph};

use std::fmt::Display;
use std::fs;
use std::time::Instant;

use clap::Parser;

use rand::Rng;

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

const N_QUERIES: usize = 10000;

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
    T
>(bvgraph: &mut BVGraph<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding, T>)
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
        + std::str::FromStr
{
    let n = bvgraph.num_nodes();

    let queries = gen_queries(N_QUERIES, n - 1);

    let mut times = Vec::default();

    let total = Instant::now();
    for &query in queries.iter() {
        let t = Instant::now();
        bvgraph.successors(query);
        times.push(t.elapsed().as_nanos());
    }
    let total = total.elapsed().as_nanos();

    println!("{:?}", times);
    println!("total: {}", total);
}

fn create_graph<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
>(props: &Properties, in_name: &str, out_name: &str)
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
        + std::str::FromStr
{
    let mut bvgraph = BVGraphBuilder::<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding, T>::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(in_name)
        .load_offsets(in_name)
        .build();

    // decompression_perf_test(&mut bvgraph);

    println!("{:?}", bvgraph.successors(6));

    // bvgraph.store2(out_name).expect("Failed storing the graph");
}

fn main() {
    let args = WGArgs::parse();

    let props = serde_json::from_str::<Properties>(
        fs::read_to_string(format!("{}.properties", args.source_name)).unwrap().as_str()
    ).unwrap();

    assert!(props.nodes as u64 <= u64::MAX, "This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", props.nodes);

    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::GAMMA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, GammaCode, usize>(&props, &args.source_name, &args.dest_name),
        (EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
            create_graph::<ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, usize>(&props, &args.source_name, &args.dest_name),
        _ => panic!("Unexpected encoding types")
        }

    // let graph = BVGraphBuilder::<GammaCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, usize>::new()
    //             .load_graph_plain(&args.source_name)
    //             .load_offsets_plain(&args.source_name)
    //             .set_window_size(args.window_size)
    //             .set_max_ref_count(args.max_ref_count)
    //             .set_min_interval_len(args.min_interval_length)
    //             .build();

    // let mut graph = BVGraphBuilder::<GammaCode, GammaCode, ZetaCode, UnaryCode, GammaCode, GammaCode, usize>::new()
    //                 .load_graph(&args.source_name)
    //                 .load_offsets(&args.source_name)
    //                 .set_window_size(args.window_size)
    //                 .set_ref_count(args.max_ref_count)
    //                 .set_min_interval_len(args.min_interval_length)
    //                 .build();

    // graph.store(&args.dest_name).expect("Failed storing the graph");
    // graph.store2(&args.dest_name).expect("Failed storing the graph");
}