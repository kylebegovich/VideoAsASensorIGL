#include "../detection_structs.h"
#include <stdio.h>

int main(){
  objectholder * object1 = malloc(sizeof(objectholder));
  objectholder * object2 = malloc(sizeof(objectholder));

  object1->x = 1;
  object1->y = 1;

  object2->x = 2;
  object2->y = 2;
  
  obj_node_t * list = init_linked_list(object1);

  push_obj(&list,object2);
  
  float test1 = get_nth_obj(1,list)->x;

  float test = pop_obj(&list)->x;

  printf("Hopefully this is 2: %f\n",test);
  printf("Hopefully this is 1: %f\n",test1);
}
