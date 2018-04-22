#ifndef DETECTION_STRUCTS_H
#define DETECTION_STRUCTS_H

#include "darknet.h"
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>

//These structs exist to create the data types needed to remember
//detections from frame to frame. 

//The general pattern is that, if the head needs to be modified,
//then the argument will be a pointer to a pointer.
//Otherwise it will just be a pointer.

int ARRAY_SIZE;

typedef int (*predicate)(int,int);

int less_than_def(int a, int b);
  

predicate less_than;




//Define a node for a linked list of objectholders.

void increase_x(objectholder** object);

obj_node_t* init_linked_list(objectholder * object);

void free_list_and_objects(obj_node_t * head);

void free_object(obj_node_t * obj);

/**
 * Structure to define a node of a linked list containing obj_node_t objects
 *  obj_node_t is a linked list of detections
 **/


/**
 * Structure to define a queue of detection_node_t objects
 * detection_node_t is a linked list of lists of detections
 **/



/** 
 * Function to append detections to queue
 * new_node->detections is null (no list of detections)
 **/

void append_detections(detections_node_t * head);

/**
 * Function to initialized detections linked list
 * All of the nodes ->detections are null
 **/

detections_node_t* init_detections_linked_list(int num_frames);

/**
 * Function to initialize detections queue
 **/

detections_queue_t* init_detections_queue(int num_frames);
/**
 * Function to add a detection to the detections queue
 **/

void add_detections(detections_node_t** head, detections_node_t** tail, int num_frames, obj_node_t* detections);


void add_detections_to_nth_frame(detections_queue_t *q, obj_node_t* detections, int n);


void add_null_detection_node(detections_queue_t * q);

void add_null_frame_nodes(detections_queue_t*** q_array);

obj_node_t * get_nth_detections(int n, detections_queue_t *q);

void push_obj(obj_node_t ** head, objectholder *object);

objectholder* pop_obj(obj_node_t ** head);
    
  


void append_obj(obj_node_t * head, objectholder * object);

int get_list_length(obj_node_t * head);

objectholder * get_nth_obj(int n, obj_node_t * head);

obj_node_t * copy_list(obj_node_t * head);



obj_node_t * map(obj_node_t * head, void (*f)(objectholder**)) ;

objectholder * copy_objectholder(objectholder obj);



detections_queue_t* ** init_global_array(int rows, int cols, int numframes);


#endif
