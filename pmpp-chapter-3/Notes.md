I make these notes to remember the details from the chapter. 

## Architecture of a modern GPU

* GPUs are organized into an array of highly threaded Streaming Multiprocessors(SMs).
* Each SM has many cores, known as CUDA cores.
* Each SM has an on-chip memory structure. 
* GPUs also come with a large amount of off-chip memory called "Global Memory."
## Block scheduling

* Threads are assigned to SMs on a block-by-block basis. All the threads in a block are assigned to the same SM simultaneously.
* Blocks need to reserve hardware resources before they can be assigned to an SM, so only a limited number of blocks can be assigned to an SM simultaneously. 
* Each GPU has a limited number of SMs, each of which can be assigned a limited number of blocks simultaneously. Hence, there is a limit to how many blocks can simultaneously run on a CUDA device. 
* The runtime maintains a list of blocks to be executed and assigns new blocks to SMs when previously assigned blocks have finished. 
* Block-by-block thread assignment allows for special interactions among threads in the same block that are not possible among threads of different blocks. This includes barrier sync and access to low-latency shared memory. 
* Threads in different blocks can also synchronize if they follow certain patterns, but that is outside the scope of this chapter. 
## Synchronization and transparent scalability

* *