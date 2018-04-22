#ifndef DETECTION_STRUCTS_H
#define DETECTION_STRUCTS_H

#include "darknet.h"
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include "detection_structs.h"

//These structs exist to create the data types needed to remember
//detections from frame to frame. 

//The general pattern is that, if the head needs to be modified,
//then the argument will be a pointer to a pointer.
//Otherwise it will just be a pointer.

int ARRAY_SIZE = 19;

typedef int (*predicate)(int,int);

int less_than_def(int a, int b) {
  return a < b;
}

predicate less_than = &less_than_def;




//Define a node for a linked list of objectholders.

void increase_x(objectholder** object) {
  (*object)->x += 5;
}

obj_node_t* init_linked_list(objectholder * object) {
  obj_node_t * new_node = NULL;
  new_node = (obj_node_t *) malloc(sizeof(obj_node_t));
  assert(new_node != NULL);
  new_node->object = object;
  new_node->next = NULL;
  
  return new_node;
  
}

void free_list_and_objects(obj_node_t * head) {
  //This will also free all the memory used for
  //the objects held in the list.

  obj_node_t * cur = head;
  obj_node_t * next_node = head->next;

  while(next_node) {
    free(cur->object);
    cur->next = NULL;
    free(cur);
    cur = next_node;
    next_node = next_node->next;
  }

  free(cur->object);
  cur->next = NULL;
  free(cur);
}
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####


void free_object(obj_node_t * obj){
  free(obj->object);
  free(obj);
}

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

void append_detections(detections_node_t * head) {
  detections_node_t * new_node = NULL;
  
  new_node = (detections_node_t *) calloc(1,sizeof(detections_node_t));
  assert(new_node != NULL);

  new_node->detections = init_linked_list(NULL);
  new_node->next = NULL;
  
  detections_node_t * current = head;
  
  while (current->next != NULL) {
    current = current->next;
    
  }
  
  current->next = new_node;

  /*
  detections_node_t * new_node =  (detections_node_t *) calloc(1, sizeof(detections_node_t));
  new_node->detections = init_linked_list(NULL);
  new_node->next = NULL;
  (*tail)->next = new_node;
  *tail = new_node;
  return *tail; */
}

/**
 * Function to initialized detections linked list
 * All of the nodes ->detections are null
 **/

detections_node_t* init_detections_linked_list(int num_frames){
  detections_node_t *  head = (detections_node_t  *) calloc(1, sizeof(detections_node_t));
  head->detections = init_linked_list(NULL);
  head->next = NULL;
 
  
  int i = 1;
  for (; i < num_frames; i++){
     append_detections(head);
  }

  return head;
}

/**
 * Function to initialize detections queue
 **/

detections_queue_t* init_detections_queue(int num_frames){
  detections_queue_t* q = (detections_queue_t*) calloc(1,sizeof(detections_queue_t));
  q->num_frames = num_frames;
  q->head = init_detections_linked_list(num_frames);
 
  detections_node_t* current = q->head;
  int i = 0;
  for (;i < num_frames-1;i++) {
    current = current->next;
  }
  q->tail = current;

  return q;
}

/**
 * Function to add a detection to the detections queue
 **/

void add_detections(detections_node_t** head, detections_node_t** tail, int num_frames, obj_node_t* detections){

  int i = 0;

  detections_node_t * new_node = (detections_node_t *) calloc(1,sizeof(detections_node_t));

  new_node->detections = detections;
  new_node->next = *head;
  *head = new_node;

  detections_node_t * prev = NULL;
  detections_node_t * iter = *head;

  while(iter != NULL){

    //prev->next implies list is longer than one

    if(iter->next == NULL){
      prev->next = NULL;
      *tail = prev;
      if(iter->detections){
        free_list_and_objects(iter->detections);
      }
      free(iter);
    }
    prev = iter;
    iter = iter->next;
  } 
}


void add_detections_to_nth_frame(detections_queue_t *q, obj_node_t* detections, int n){

  detections_node_t** head = &q->head;
  detections_node_t** tail = &q->tail;
  int num_frames = q->num_frames;
 
  int i = 0;
  detections_node_t * iter = *head;

  while(iter != NULL){

    if(i == n ){
	iter->detections = detections;
    }

    iter = iter->next;
    i++;
  } 
}


void add_null_detection_node(detections_queue_t * q){

  //  detections_node_t** head = &q.head;
  //detections_node_t** tail = &q.tail;
  int num_frames = q->num_frames;

  int i = 0;

  detections_node_t * new_node = (detections_node_t *) calloc(1,sizeof(detections_node_t));

  new_node->detections = init_linked_list(NULL);
  new_node->next = q->head;
  q->head = new_node;

  detections_node_t * prev = NULL;
  detections_node_t * iter = q->head;

  while(iter->next != NULL){

    //prev->next implies list is longer than one
    prev = iter;
    iter = iter->next;


  }
  prev->next = NULL;
  q->tail = prev; 
  free_list_and_objects(iter->detections);
  free(iter);
}

void add_null_frame_nodes(detections_queue_t*** q_array){

  int i,j;
  for(i = 0; i < ARRAY_SIZE; i++){
    for(j = 0; j < ARRAY_SIZE; j++) {
      add_null_detection_node(q_array[i][j]);
    }
  }
}

obj_node_t * get_nth_detections(int n, detections_queue_t *q) {

  if ((n < 0) || (n >= q->num_frames)){
    return NULL;
  }

  //Indexing starts from 0, we'll allow negative numbers like python.
  obj_node_t * to_return = NULL;
  detections_node_t * current = q->head;
  int i = 0;
    
 
  for (; i < n; i++) {
    current = current->next;
  }

  to_return =  current->detections;
   
  return to_return; 
}




//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####
//####FELIPE####




void push_obj(obj_node_t ** head, objectholder *object) {
  obj_node_t * new_node = NULL;
  new_node = (obj_node_t *) malloc(sizeof(obj_node_t));
  assert(new_node != NULL);
  new_node->object = object;
  new_node->next = *head;
  *head = new_node;
}

objectholder* pop_obj(obj_node_t ** head) {
  objectholder * return_val = (*head)->object;

  if ((*head)->next == NULL) {
    free(*head);
    return return_val;
  } else {
    obj_node_t * headcopy = *head;
    *head = (*head)->next;
    free(headcopy);
    return return_val;
  }
    
  
}

void append_obj(obj_node_t * head, objectholder * object) {
  obj_node_t * new_node = NULL;
  
  new_node = (obj_node_t *) malloc(sizeof(obj_node_t));
  assert(new_node != NULL);

  new_node->object = object;
  new_node->next = NULL;
  
  obj_node_t * current = head;
  
  while (current->next != NULL) {
    current = current->next;
    
  }
  
  current->next = new_node;
}

int get_list_length(obj_node_t * head) {
  //No empty lists allowed.
  int len = 1;
  obj_node_t * current = head;
  while(current->next != NULL) {
    current = current->next;
    len++;
  }
  return len;
}

objectholder * get_nth_obj(int n, obj_node_t * head) {
  //Indexing starts from 0, we'll allow negative numbers like python.
  objectholder * to_return = NULL;
  obj_node_t * current = head;
  int length = get_list_length(head);
  
  assert((n >= 0 && n < length) || (n < 0 && 0 - n < length));

  if (n < 0) n = length + n;
  
  int current_obj = 0;

  while (current_obj < n) {
    current = current->next;
    current_obj++;
  }

  return current->object; 
}

obj_node_t * copy_list(obj_node_t * head) {
  obj_node_t * copy = NULL;
  obj_node_t **tail = &copy;
  objectholder * temp_obj;
  for(;head; head = head->next) {
    *tail = (obj_node_t *) malloc(sizeof(obj_node_t));
    temp_obj = (objectholder *) malloc(sizeof(objectholder));
    
    *temp_obj = *(head->object);
    (*tail)->object = temp_obj;
    (*tail)->next = NULL;
    tail = &((*tail)->next);
  }
  return copy;
}



obj_node_t * map(obj_node_t * head, void (*f)(objectholder**)) {
  obj_node_t * return_list = copy_list(head);
  obj_node_t ** tail = &return_list;
  
  for(;head; head = head->next) {
    (*f)(&((*tail)->object));
    tail = &((*tail)->next);
  }
  
  return return_list;
}

objectholder * copy_objectholder(objectholder obj) {
  objectholder * to_return = (objectholder *) malloc(sizeof(objectholder));
  to_return->x = obj.x;
  to_return->y = obj.y;
  to_return->w = obj.w;
  to_return->h = obj.h;
  to_return->confidence = obj.confidence;
  to_return->name = obj.name;
  to_return->index = obj.index;
  to_return->is_compound_object = obj.is_compound_object;
  to_return->object1_index = obj.object1_index;
  to_return->object2_index = obj.object2_index;
  to_return->future = obj.future;
  return to_return;
}



detections_queue_t* ** init_global_array(int rows, int cols, int numframes) {
  int i,j;

  //Array of pointers to detection queues
  detections_queue_t* * data = calloc(rows*cols,sizeof(detections_queue_t*));
  
  //Double array of pointers to detection queues 
   detections_queue_t* ** to_return = calloc(rows,sizeof(detections_queue_t**));
   for (i = 0; i < rows; i++) {
     to_return[i] = data + (i * cols);
   }
   
   for (i = 0; i < rows; i++) {
     for (j = 0; j < cols; j++) {
      to_return[i][j] = init_detections_queue(numframes);
    }
  }

  return to_return;
}


#endif
