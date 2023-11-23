use webgraph_rust::ascii_graph::{AsciiGraphBuilder, AsciiGraph};
use webgraph_rust::properties::Properties;
use webgraph_rust::utils::EncodingType;
use webgraph_rust::utils::encodings::{UniversalCode, GammaCode, UnaryCode, ZetaCode, DeltaCode};
use webgraph_rust::webgraph::bvgraph::{BVGraph, BVGraphBuilder};
use webgraph_rust::ImmutableGraph;

use core::panic;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use clap::Parser;

use rand::Rng;

#[derive(Parser, Debug)]
struct WGArgs {
    /// The size of the window
    #[arg(short = 'w', long = "window-size", default_value_t = 7)]
    window_size: usize,
    /// The maximum reference chain length
    #[arg(short = 'r', long = "max-ref-count", default_value_t = 3)]
    max_ref_count: usize,
    /// The minimum length of the interval
    #[arg(short = 'i', long = "min-interval-len", default_value_t = 4)]
    min_interval_length: usize,
    /// Specifies the block coding type
    #[arg(long = "block-coding", default_value_t = EncodingType::GAMMA)]
    block_coding: EncodingType,
    /// Specifies the block count coding type
    #[arg(long = "block-count-coding", default_value_t = EncodingType::GAMMA)]
    block_count_coding: EncodingType,
    /// Specifies the outdegree coding type
    #[arg(long = "outdegree-coding", default_value_t = EncodingType::GAMMA)]
    outdegree_coding: EncodingType,
    /// Specifies the offset coding type
    #[arg(long = "offset-coding", default_value_t = EncodingType::GAMMA)]
    offset_coding: EncodingType,
    /// Specifies the reference coding type
    #[arg(long = "reference-coding", default_value_t = EncodingType::UNARY)]
    reference_coding: EncodingType,
    /// Specifies the intervals coding type
    #[arg(long = "interval-coding", default_value_t = EncodingType::GAMMA)]
    interval_coding: EncodingType,
    /// Specifies the residual coding type
    #[arg(long = "residual-coding", default_value_t = EncodingType::ZETA)]
    residual_coding: EncodingType,
    /// Specifies the k parameter for ZetaK coding
    #[arg(short = 'k', long = "zetak", default_value_t = 3)]
    zeta_k: usize,
    /// Whether to compress offsets through Elias-Fano. It overwrites the offset_coding, if specified.
    #[arg(long = "ef", default_value_t = false)]
    elias_fano: bool,
    /// Source basename
    source_name: String,                                                    // TODO: PathBuf instead of str?
    /// Destination basename
    dest_name: Option<String>,                                              // TODO: PathBuf instead of str?
    /// Check that the written graph is equal to the read graph
    #[arg(short, long = "check", default_value_t = false)]
    check: bool,
    /// Performance test
    #[arg(short, long = "perf", default_value_t = false)]
    perf_test: bool,
    /// Compress starting from an ASCII graph
    #[arg(short = 'a', long = "ascii", default_value_t = false)]
    from_plain: bool,
}

const N_QUERIES: usize = 1000000;

fn gen_queries(n_queries: usize, range_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..n_queries)
        .map(|_| rng.gen_range(0..range_size))
        .collect()
}

fn decompression_perf_test<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutDegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutDegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
>(bvgraph: &mut BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutDegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutDegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
>) {
    let n = bvgraph.num_nodes();

    let queries = gen_queries(N_QUERIES, n - 1);

    let total = Instant::now();
    for &query in queries.iter() {
        bvgraph.decode_list(query, &mut bvgraph.graph_binary_wrapper.borrow_mut(), None, &mut []);
    }
    let avg_query = (total.elapsed().as_nanos() as f64) / N_QUERIES as f64;

    println!("time per query: {}ns", avg_query);
}

fn create_graph<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutDegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutDegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
>(props: &Properties, args: &WGArgs, plain_graph: Option<AsciiGraph<usize>>) {
    if let Some(plain_graph) = plain_graph {
        let bvgraph = BVGraphBuilder::<
            InBlockCoding,
            InBlockCountCoding,
            InOutDegreeCoding,
            InOffsetCoding,
            InReferenceCoding,
            InIntervalCoding,
            InResidualCoding,
            OutBlockCoding,
            OutBlockCountCoding,
            OutOutDegreeCoding,
            OutOffsetCoding,
            OutReferenceCoding,
            OutIntervalCoding,
            OutResidualCoding,
        >::new()
            .set_out_min_interval_len(args.min_interval_length)
            .set_out_max_ref_count(args.max_ref_count)
            .set_out_window_size(args.window_size)
            .set_out_zeta(Some(args.zeta_k as u64))
            .set_num_nodes(plain_graph.num_nodes())
            .set_num_edges(plain_graph.num_arcs())
            .build();

        let comp_time = Instant::now();
        bvgraph.store_plain(&plain_graph, args.dest_name.as_ref().unwrap().as_str()).expect("Failed storing the plain graph");
        let comp_time = comp_time.elapsed().as_nanos() as f64;
        println!("compressed the plain graph in {}ns", comp_time);
    } else {
        let mut bvgraph = BVGraphBuilder::<
            InBlockCoding,
            InBlockCountCoding,
            InOutDegreeCoding,
            InOffsetCoding,
            InReferenceCoding,
            InIntervalCoding,
            InResidualCoding,
            OutBlockCoding,
            OutBlockCountCoding,
            OutOutDegreeCoding,
            OutOffsetCoding,
            OutReferenceCoding,
            OutIntervalCoding,
            OutResidualCoding,
        >::new()
            .set_in_min_interval_len(props.min_interval_len)
            .set_out_min_interval_len(args.min_interval_length)
            .set_in_max_ref_count(props.max_ref_count)
            .set_out_max_ref_count(args.max_ref_count)
            .set_in_window_size(props.window_size)
            .set_out_window_size(args.window_size)
            .set_in_zeta(props.zeta_k)
            .set_out_zeta(Some(args.zeta_k as u64))
            .set_elias_fano(args.elias_fano)
            .set_num_nodes(props.nodes)
            .set_num_edges(props.arcs)
            .load_graph(&args.source_name)
            .load_offsets(&args.source_name)
            .load_outdegrees()
            .build();

        if args.perf_test {
            decompression_perf_test(&mut bvgraph);
        } else if let Some(out_name) = args.dest_name.as_ref(){
            let comp_time = Instant::now();
            bvgraph.store(out_name.as_str()).expect("Failed storing the graph");
            let comp_time = comp_time.elapsed().as_nanos() as f64;
            println!("compressed the graph in {}ns", comp_time);

            if args.check {
                let compressed_graph = BVGraphBuilder::<
                    InBlockCoding,
                    InBlockCountCoding,
                    InOutDegreeCoding,
                    InOffsetCoding,
                    InReferenceCoding,
                    InIntervalCoding,
                    InResidualCoding,
                    OutBlockCoding,
                    OutBlockCountCoding,
                    OutOutDegreeCoding,
                    OutOffsetCoding,
                    OutReferenceCoding,
                    OutIntervalCoding,
                    OutResidualCoding,
                >::new()
                    .set_in_min_interval_len(args.min_interval_length)
                    .set_in_max_ref_count(args.max_ref_count)
                    .set_in_window_size(args.window_size)
                    .set_in_zeta(props.zeta_k)
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

    if args.from_plain && args.dest_name.is_none() {
        panic!("No destination name provided.");
    }

    let mut plain_graph = None;
    let mut props = Properties::default();
    
    if !args.from_plain {
        let properties_file = File::open(format!("{}.properties", args.source_name));
        let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
        let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));

        props = Properties::from(p);
    } else {
        plain_graph = Some(AsciiGraphBuilder::new()
                        .load_ascii(&args.source_name)
                        .build());
    }
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding, 
        args.block_coding, args.block_count_coding, args.outdegree_coding, args.offset_coding, args.reference_coding, args.interval_coding, args.residual_coding) {
    (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA, // Default case
    EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA) => 
        create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode>
        (&props, &args, plain_graph),
    (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA, // Default to gamma
    EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
        create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode>
        (&props, &args, plain_graph),
    (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA, // Default to delta
    EncodingType::DELTA, EncodingType::DELTA, EncodingType::DELTA, EncodingType::DELTA, EncodingType::DELTA, EncodingType::DELTA, EncodingType::DELTA) => 
        create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode, DeltaCode, DeltaCode, DeltaCode, DeltaCode, DeltaCode, DeltaCode, DeltaCode>
        (&props, &args, plain_graph),
    (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA, // Default to zeta
    EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA, EncodingType::ZETA) => 
        create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode, ZetaCode>
        (&props, &args, plain_graph),
    _ => panic!("Unexpected encoding types", )
}
}