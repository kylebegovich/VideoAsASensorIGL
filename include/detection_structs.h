#ifndef DETECTION_STRUCTS_H
#define DETECTION_STRUCTS_H

#include <unistd.h>
#include <stdlib.h>
#include <assert.h>

//These structs exist to create the data types needed to remember
//detections from frame to frame. 

//The general pattern is that, if the head needs to be modified,
//then the argument will be a pointer to a pointer.
//Otherwise it will just be a pointer.


typedef int (*predicate)(int,int);

int less_than_def(int a, int b) {
  return a < b;
}

predicate less_than = &less_than_def;


typedef struct objectholder{

  float x; //(x,y) represent the coordinates of the centroid
  float y; // of the detection.
  float w; // w,h are the relative width and height of the bounding box
  float h; 
  float confidence;
  char * name;
  int index; //Remembers the index in an original list that an object came from.
  int is_compound_object;
 struct  objectholder* object1_index; //For compound objects -- helps provide the projection function compound_object_list \subset object1_list x object2_list --> object1_list.
  struct objecthodler* object2_index; //similar to above.
  // This is just for convenience in a few functions later on, and won't be displayed.

} objectholder;

//Define a node for a linked list of objectholders.
typedef struct obj_node_t {
  objectholder * object;
  struct obj_node_t * next;
} obj_node_t;

void increase_x(objectholder** object) {
  (*object)->x += 5;
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
typedef struct detections_node_t{
obj_node_t * detections;
struct detections_node_t* next;

} detections_node_t;

/**
 * Structure to define a queue of detection_node_t objects
 * detection_node_t is a linked list of lists of detections
 **/

typedef struct detections_queue_t{
int num_frames;
detections_node_t* head;
detections_node_t* tail;

} detections_queue_t;

/** 
 * Function to append detections to queue
 * new_node->detections is null (no list of detections)
 **/

detections_node_t* append_detections(detections_node_t ** tail) {
  detections_node_t * new_node =  (detections_node_t *) calloc(1, sizeof(detections_node_t));
  (*tail)->next = new_node;
  *tail = new_node;
  return *tail;
}

/**
 * Function to initialized detections linked list
 * All of the nodes ->detections are null
 **/

detections_node_t* init_detections_linked_list(int num_frames, detections_node_t** tail){
  detections_node_t *  head = (detections_node_t  *) calloc(1, sizeof(detections_node_t));
  (*tail) = head;

  int i = 1;
  for (; i < num_frames; i++){
    *(tail) = append_detections(tail);
  }

  return head;
}

/**
 * Function to initialize detections queue
 **/

detections_queue_t* init_detections_queue(int num_frames){
  detections_queue_t* q = (detections_queue_t*) calloc(1,sizeof(detections_queue_t));
  q->head = init_detections_linked_list(num_frames, &q->tail);
  q->num_frames = num_frames;

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
        free_object(iter->detections);
      }
      free(iter);
    }
    prev = iter;
    iter = iter->next;
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
    
  while (current != NULL) {
    if(i == n){
      return current->detections;
    }
    current = current->next;
    i++;
  }

  return NULL; 
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


obj_node_t* init_linked_list(objectholder * object) {
  obj_node_t * new_node = NULL;
  new_node = malloc(sizeof(obj_node_t));
  assert(new_node != NULL);
  new_node->object = object;
  new_node->next = NULL;
  
  return new_node;
  
}

void push_obj(obj_node_t ** head, objectholder *object) {
  obj_node_t * new_node = NULL;
  new_node = malloc(sizeof(obj_node_t));
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
  
  new_node = malloc(sizeof(obj_node_t));
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
    *tail = malloc(sizeof(obj_node_t));
    temp_obj = malloc(sizeof(objectholder));
    
    *temp_obj = *(head->object);
    (*tail)->object = temp_obj;
    (*tail)->next = NULL;
    tail = &((*tail)->next);
  }
  return copy;
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
  objectholder * to_return = malloc(sizeof(objectholder));
  to_return->x = obj.x;
  to_return->y = obj.y;
  to_return->w = obj.w;
  to_return->h = obj.h;
  to_return->confidence = obj.confidence;
  to_return->name = obj.name;
  to_return->index = obj.index;
  to_return->is_compound_object = obj.is_compound_object;

  return to_return;
}

void init_global_array(detections_queue_t ** array_to_init, int rows, int cols, int numframes) {
  int i,j;
  for (i = 0; i < rows; i++) {
    for (j = 0; i < cols; j++) {
      array_to_init[i][j] = *init_detections_queue(numframes);
    }
  }
}


#endif
