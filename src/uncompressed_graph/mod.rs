use std::{fs, str::FromStr, fmt};

use crate::ImmutableGraph;

struct UncompressedGraph<T> {
    n: usize,
    m: usize,
    graph_memory: Vec<T>,
    offsets: Vec<usize>,
}

impl<T> ImmutableGraph for UncompressedGraph<T> 
where T: PartialOrd<i32> 
        + PartialOrd<usize> 
        + num_traits::PrimInt 
{
    type NodeT = T;

    fn num_nodes(&self) -> usize {
        self.n
    }

    fn num_arcs(&self) -> usize {
        self.m
    }

    /// Returns the number of adjacent nodes of the given node (if in bounds).
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number (from 0 to n)
    fn outdegree(&self, x: Self::NodeT) -> Option<usize> {
        // offsets are not correlated with the number of nodes since a node
        // can have no outgoing edges, thus it is not represented in graph_memory   // TODO: not true
        if x < 0 || x > self.offsets.len() {
            return None;
        }
        
        let usized_ind = x.to_usize().unwrap();
        let left_index = self.offsets[usized_ind];

        if left_index < self.offsets.len() - 1 {
            let right_index = self.offsets[usized_ind + 1];
            return Some(right_index - left_index - 1);
        } 

        // If it is the last node having any successor, then
        // return the remainder of the graph_memory nodes list        
        Some(self.graph_memory.len() - left_index - 1)
    }

    fn successors(&self, x: Self::NodeT) -> Result<Box<dyn Iterator<Item = &u32>>, &str> {
        todo!()
    }    
}


struct ImmutableGraphBuilder<T> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<T>,
    loaded_offsets: Vec<usize>,
}

impl<T> ImmutableGraphBuilder<T>
where 
    T: num_traits::PrimInt + FromStr,
    <T as FromStr>::Err: fmt::Debug 
{
    fn new() -> ImmutableGraphBuilder<T> {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::new(), 
            loaded_offsets: Vec::new() 
        }
    }

    fn load_graph(&mut self, filename: &str) {
        self.loaded_graph = fs::read_to_string(filename)
                            .expect("Failed to load the graph file")
                            .split(" ")
                            .map(|node| node.parse()
                                                    .expect(format!("Failed to parse node {}", node).as_str())
                            )
                            .collect();        
    }

    fn load_offsets(&mut self, filename: &str) {
        self.loaded_offsets = fs::read_to_string(filename)
                            .expect("Failed to load the offsets file")
                            .split(" ")
                            .map(|node| node.parse()
                                                    .expect(format!("Failed to parse offset {}", node).as_str())
                            )
                            .collect();        
    }


    /// Computes the number of nodes by counting the offsets' file entries.
    /// 
    /// This is correct since all the nodes are represented in the graph file, even those
    /// not having any successor. Hence, their positions are written in the offsets file,
    /// and the amount of entries corresponds to the total amount of nodes. 
    fn count_nodes(&mut self) {
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_nodes = self.loaded_offsets.len();
    }

    fn count_arcs(&mut self) {
        assert!(!self.loaded_graph.is_empty(), "The graph has to be loaded");

        todo!();
    }

    /// Constructs the UncompressedGraph object.
    fn construct(self) -> UncompressedGraph<T> {
        UncompressedGraph { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph, 
            offsets: self.loaded_offsets 
        }
    }
}
