use crate::ImmutableGraph;

struct UncompressedGraph<T> {
    n: usize,
    m: usize,
    graph_memory: Vec<T>,
    offsets: Vec<usize>,
}

impl<T> ImmutableGraph for UncompressedGraph<T> where T: PartialOrd<i32> + PartialOrd<usize> + num_traits::PrimInt {
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
        // can have no outgoing edges, thus it is not represented in graph_memory
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


struct ImmutableGraphBuilder {
    
}
