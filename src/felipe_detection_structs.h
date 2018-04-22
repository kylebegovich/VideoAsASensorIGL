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

//Define a node for a linked list of obj_node_ts

typedef struct detections_node_t{

obj_node_t * detections;
struct detections_node_t* next;

} detections_node_t;


detections_node_t* init_detections_linked_list(obj_node_t * detections){
  detections_node_t *  new_node = NULL;
  new_node = (detections_node_t  *)malloc(sizeof(detections_node_t));
  new_node->detections = detections;
  new_node->next = NULL;
  
  return new_node;
}

void* push_detections(detections_node_t ** head, obj_node_t * detections) {
  detections_node_t * new_node = NULL;
  new_node = (detections_node_t *) malloc(sizeof(detections_node_t));
  new_node->detections = detections;
  new_node->next = *head;
  *head = new_node;
}


//####FELIPE####



obj_node_t * init_linked_list(objectholder * object) {
  obj_node_t * new_node = NULL;
  new_node = malloc(sizeof(obj_node_t));
  assert(new_node != NULL);
  new_node->object = object;
  new_node->next = NULL;
  
  return new_node;
  
}

void* push_obj(obj_node_t ** head, objectholder *object) {
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

void* append_obj(obj_node_t * head, objectholder * object) {
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
    free(cur);
    cur = next_node;
    next_node = next_node->next;
  }

  free(cur->object);
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

#endif
