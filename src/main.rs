use std::{env, process};

use webgraph_rust::webgraph::{EncodingType, BVGraph};

fn main() {
    if env::args().len() != 9 {
        println!("Wrong arguments. The correct usage is {}\n
            \tcompression-type: (1-7)\n
                \t\tDelta: 1\n
                \t\tGamma: 2\n
                \t\tGolomb: 3\n
                \t\tSkewed Golomb: 4\n
                \t\tUnary: 5\n
                \t\tZeta: 6\n
                \t\tNibble: 7\n
            \twindow-size: int\n
            \tmax-ref-count: int\n
            \tmin-interval-length: int\n
            \tzetak: int\n
            \tstore-outdeg: 0 (true) / 1 (false)\n
            \tsource-name: string\n
            \dest-name: string\n",
            env::args().into_iter().next().unwrap()
        );
        process::exit(0);
    }

    let mut arguments = env::args().into_iter();

    let comp = match arguments.next()?.parse() {
        Some(1) => EncodingType::DELTA,
        Some(2) => EncodingType::GAMMA,
        Some(3) => EncodingType::GOLOMB,
        Some(4) => EncodingType::SKEWEDGOLOMB,
        Some(5) => EncodingType::UNARY,
        Some(6) => EncodingType::ZETA,
        Some(7) => EncodingType::NIBBLE,
        _ => EncodingType::ZETA, // Default compression is zetak
    };

    arguments.next();

    let window_size = arguments.next()?.parse().unwrap();
    let window_size = if window_size >= 0 {window_size} else {0}; // 0 = not used

    let max_ref_count = arguments.next()?.parse().unwrap();
    let max_ref_count = if max_ref_count >= 0 {max_ref_count} else {0}; // 0 = not used

    let min_interval_length = arguments.next()?.parse().unwrap();
    let min_interval_length = if min_interval_length >= 2 {min_interval_length} else {0}; // Interval len is either 2 either not used (0)

    let zetak = arguments.next()?.parse().unwrap();
    let zetak = if comp == EncodingType::ZETA && zetak >= 0 {zetak} else if EncodingType::ZETA {3} else {0}; // 3 is the default k value, 0 = not used

    // Probably can be removed
    let store_outdeg = arguments.next()?.parse().unwrap();
    let store_outdeg = if store_outdeg == 0 {0} else {1}; // 0 as default
    
    let source_name = arguments.next()?;

    let dest_name = arguments.next()?;

    let graph = BVGraph::load(&source_name); // Loading the graph

    // ...ops...

    graph.store(&dest_name); // Storing the graph

}