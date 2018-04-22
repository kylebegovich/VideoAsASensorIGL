#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>
#include "detection_structs.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define IMAGE 0  //Set to run on images instead of videos. Will not save anything with openCV on.
#define SAVEIMAGE 0
#define LIVEVIDEO 1
#define SAVEVIDEO 0
#define TRAJECTORYWARNINGS 0 //Set to toggle cautions signs for people near middle of screen.
#define BLUR 0 //Set to toggle blurring of people boxes
#define ONLYBIKERS 1 //Set to display only bikes, people, and biker (person on a bike) detections.
#define BIKERS 1
#define SAVEJSON 1 //Set to save a json file containing bike/people detections.
#define LANE_DETECTION 0
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static FRAME_NUM = 0;

#if LANE_DETECTION
enum{
    CANNY_MIN_TRESHOLD = 100,	  // edge detector minimum hysteresis threshold
	CANNY_MAX_TRESHOLD = 150, // edge detector maximum hysteresis threshold

	HOUGH_TRESHOLD = 120,		// line approval vote threshold
	HOUGH_MIN_LINE_LENGTH = 130,	// remove lines shorter than this treshold
	HOUGH_MAX_LINE_GAP = 150,   // join lines to one with smaller than this gaps
};
#endif


int get_grid_x(objectholder * obj) {
  return  ((int) ( obj->x * 19 )) % 19;
}

int get_grid_y(objectholder * obj) {
  return ((int) ( obj->y * 19 )) % 19;
}

void displayobjecthold(objectholder *object, FILE * filepointer,int comma) {
  if (comma) {
    fprintf(filepointer,"{\n \"x\" : %.3f, \n \"y\":%.3f, \n \"confidence\" : %.3f, \n \"name\": \"%s\", \n \"w\" : %.3f, \n \"h\": %.3f, \n \"is_compound_object\" : %d \n},",object->x,object->y,object->confidence,object->name,object->w,object->h,object->is_compound_object);
  } else {
    fprintf(filepointer,"{\n \"x\" : %.3f, \n \"y\":%.3f, \n \"confidence\" : %.3f, \n \"name\": \"%s\", \n \"w\" : %.3f, \n \"h\" : %.3f, \n \"is_compound_object\" : %d \n } \n",object->x,object->y,object->confidence,object->name,object->w,object->h,object->is_compound_object);
  }
  return;
}



void writedetectionlist_non_compound(objectholder * firstlist, objectholder * secondlist,int * matchlist, int firstcount, int secondcount, FILE * fp, char * classtype){
  int i,j;
  const int COMMA = 1;
  const int NOCOMMA = 0;
  fprintf(fp,"{\n \"type\" : \"%s\", \n \"detections\" : [\n",classtype);
  for (i = 0; i < firstcount; ++i) {
    for (j = 0; j < secondcount; ++j) {
      if (j != matchlist[i]) {
	if(secondcount-1 == matchlist[firstcount-1]) {
	  if(i != firstcount - 1 || j != secondcount- 2) {
	fprintf(fp,"[\n"); displayobjecthold(&firstlist[i],fp,COMMA); fprintf(fp,"\n"); displayobjecthold(&secondlist[matchlist[j]],fp,NOCOMMA); fprintf(fp,"],");
      } else {
	fprintf(fp,"[\n"); displayobjecthold(&firstlist[i],fp,COMMA); fprintf(fp,"\n"); displayobjecthold(&secondlist[matchlist[j]],fp,NOCOMMA); fprintf(fp,"]");
      }
	} else
	  if(i != firstcount - 1 || j != secondcount- 1) {
	       fprintf(fp,"[\n"); displayobjecthold(&firstlist[i],fp,COMMA); fprintf(fp,"\n"); displayobjecthold(&secondlist[matchlist[j]],fp,NOCOMMA); fprintf(fp,"],");
	     } else {
	       fprintf(fp,"[\n"); displayobjecthold(&firstlist[i],fp,COMMA); fprintf(fp,"\n"); displayobjecthold(&secondlist[matchlist[j]],fp,NOCOMMA); fprintf(fp,"]");
	  }
      }
    }
  }
  fprintf(fp, "]}\n");
}

/*
matchlist is a function f: firstobjectlist ---> secondobjectlist
which pairs up each component of a compound object with 
its complementary object.

The function is defined as f(firstobjectlist[i]) = matchlist[i]
*/

void writedetectionlist_compound(objectholder * firstlist, objectholder * secondlist,int * matchlist, int objectcount, FILE * fp, char * classtype){
  int i;
  const int COMMA = 1;
  const int NOCOMMA = 0;
  fprintf(fp,"{\n \"type\" : \"%s\", \n \"detections\" : [\n",classtype);
  for (i = 0; i < objectcount; ++i) {
    if (firstlist[i].is_compound_object == 1) {
      if(i != objectcount - 1) {
	fprintf(fp,"[\n"); displayobjecthold(&firstlist[i],fp,COMMA); fprintf(fp,"\n"); displayobjecthold(&secondlist[matchlist[i]],fp,NOCOMMA); fprintf(fp,"],");
      } else {
	fprintf(fp,"[\n"); displayobjecthold(&firstlist[i],fp,COMMA); fprintf(fp,"\n"); displayobjecthold(&secondlist[matchlist[i]],fp,NOCOMMA); fprintf(fp,"]");
      }
    }
  }
  fprintf(fp, "]}\n");
}


box objectholder_to_box(objectholder object)
{
    box b;
    b.x = object.x;
    b.y = object.y;
    b.w = object.w;
    b.h = object.h;
    return b;
}

int is_on_top(box top, box bot){
  return top.y - top.h/2. <= bot.y + bot.h/2.;
}

int is_stacked_object(objectholder top_obj, objectholder bot_obj){
    box box_1 = objectholder_to_box(top_obj);
    box box_2 = objectholder_to_box(bot_obj);
    float IOU = box_iou(box_1, box_2);

    if(IOU >= 0.1 && is_on_top(box_1,box_2)){
        return 1;
    }

    return 0;
}

float get_euclidean_distance(objectholder object_1, objectholder object_2){
    return (object_1.x-object_2.x)*(object_1.x-object_2.x) + (object_1.y-object_2.y)*(object_1.y - object_2.y);
}



int windows = 0;
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

/**** COMPOUND OBJECT LOGIC ***/
////////////////////////////////

void check_grid_cell_for_past(int framenum, int x, int y, float * max_IOU, objectholder * to_check , objectholder ** cur_past, detections_queue_t* ** DETECTIONS_ARRAY) {
  
  obj_node_t * current = NULL;

  if (strncmp(to_check->name,"person",strlen("person")) == 0) {
    current = get_nth_detections(framenum+1,DETECTIONS_ARRAY[x][y]);

  } else  if (strncmp(to_check->name,"bicycle",strlen("bicycle")) == 0) {
    current = get_nth_detections(framenum+1,DETECTIONS_ARRAY[x][y]);

  } else {
    printf("Invalid type. \n");
    return;
  }

  float test_IOU = -1;

  while (current) {
    if (current->object != NULL) {
      test_IOU = box_iou(objectholder_to_box(*current->object),objectholder_to_box(*to_check)); 
      if (test_IOU > *max_IOU) {
	*max_IOU = test_IOU;
	*cur_past = current->object;
      }
    }
    current = current->next;
  }
}

void find_past_object (int framenum, objectholder * obj, int x, int y, detections_queue_t* ** DETECTIONS_ARRAY) {
  //Conditions to test to find the past object. 
  //Assumptions: probably have to assume high framerate video
  //Round 1: search for highest IOU object.

  //We don't want to be finding a past object for things in the last frame:
  int MAX_FRAMES = 5;
  assert(framenum < MAX_FRAMES - 1);


  //Set up variables
  objectholder * cur_past = NULL;
  float max_IOU = -1;      
  

  //For a first iteration, we were just checking gridcell x and y. We should really check all adjacent grid cells :(. We'll start by doing 4-adjancency, and add in 8-adjacency after.


    //Check center cell
    check_grid_cell_for_past(framenum,x,y, &max_IOU, obj, &cur_past, DETECTIONS_ARRAY);
    //Check left cell
    if (x > 0) {
      check_grid_cell_for_past(framenum,x-1,y, &max_IOU, obj, &cur_past, DETECTIONS_ARRAY);
    }
    //Check right cell, grid range is 0-18 currently
    if (x < 18) {
      check_grid_cell_for_past(framenum,x+1,y, &max_IOU, obj, &cur_past, DETECTIONS_ARRAY);
    }
    //Check top cell
    if (y > 0) {
       check_grid_cell_for_past(framenum,x,y-1, &max_IOU, obj, &cur_past, DETECTIONS_ARRAY);
    }
    //Check bottom cell
    if (y < 18) {
       check_grid_cell_for_past(framenum,x,y+1, &max_IOU, obj, &cur_past, DETECTIONS_ARRAY);
    }
    //Done checking, update the past
    obj->past = cur_past;

    if (obj->past != NULL)  {
      obj->past->future = obj;
    }
  
}


int test_compound_object (objectholder * objtop, objectholder * objbot) {
  if (is_stacked_object(*objtop,*objbot)) {
   
    return 1;
  } 
   if (objtop->past != NULL){ 
    if (objtop->past->is_compound_object == 1) {
      return 1;
      if (objtop->past->past != NULL) {
	if (objtop->past->past->is_compound_object == 1) {
	  return 1;
	}
      }
    
    }
   }
    
   
   return 0;
}

/*** END COMPOUND OBJECT LOGIC ***/



image mask_to_rgb(image mask)
{
    int n = mask.c;
    image im = make_image(mask.w, mask.h, 3);
    int i, j;
    for(j = 0; j < n; ++j){
        int offset = j*123457 % n;
        float red = get_color(2,offset,n);
        float green = get_color(1,offset,n);
        float blue = get_color(0,offset,n);
        for(i = 0; i < im.w*im.h; ++i){
            im.data[i + 0*im.w*im.h] += mask.data[j*im.h*im.w + i]*red;
            im.data[i + 1*im.w*im.h] += mask.data[j*im.h*im.w + i]*green;
            im.data[i + 2*im.w*im.h] += mask.data[j*im.h*im.w + i]*blue;
        }
    }
    return im;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if(c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

static float bilinear_interpolate(image im, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
    return val;
}


void composite_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}

image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k){
        for(y = 0; y < b.h; ++y){
            for(x = 0; x < b.w; ++x){
                float val = get_pixel_extend(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}

image tile_images(image a, image b, int dx)
{
    if(a.w == 0) return copy_image(b);
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0); 
    composite_image(b, c, a.w + dx, 0);
    return c;
}

image get_label(image **characters, char *string, int size)
{
    if(size > 7) size = 7;
    image label = make_empty_image(0,0,0);
    while(*string){
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_fill_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
  int i,j;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
      for (j = y1; j <= y2; ++j) {
        a.data[i + j*a.w + 0*a.w*a.h] = r;
   

        a.data[i + j*a.w + 1*a.w*a.h] = g;
   

        a.data[i + j*a.w + 2*a.w*a.h] = b;
   
      }
    }
   
}

image sub_image(image a, int x_min, int y_min, int x_max, int y_max)
{
    //Look for edge cases
    int i,j;
    if(x_min < 0) x_min = 0;
    if(x_min >= a.w) x_min = a.w-1;
    if(x_max < 0) x_max = 0;
    if(x_max >= a.w) x_max = a.w-1;

    if(y_min < 0) y_min = 0;
    if(y_min >= a.h) y_min = a.h-1;
    if(y_max < 0) y_max = 0;
    if(y_max >= a.h) y_max = a.h-1;
    
    int new_image_width = x_max - x_min + 1;
    int new_image_height = y_max - y_min + 1;


    int new_image_size = a.c*(new_image_width)*(new_image_height);
    float *sub_image_data = malloc(new_image_size*sizeof(float));


    for(i = 0; i <= new_image_width-1; ++i){
      for (j = 0; j <= new_image_height-1; ++j) {
        sub_image_data[i + j*new_image_width+0*new_image_width*new_image_height] = a.data[i+x_min + (j+y_min)*a.w + 0*a.w*a.h]; 
	sub_image_data[i + j*new_image_width+1*new_image_width*new_image_height] = a.data[i + x_min + (j+y_min)*a.w + 1*a.w*a.h]; 
	sub_image_data[i + j*new_image_width+2*new_image_width*new_image_height] = a.data[i+x_min + (j+y_min)*a.w + 2*a.w*a.h]; 
      }
    }

    image sub_image = float_to_image(new_image_width, new_image_height, a.c, sub_image_data);
    
    return sub_image;
}

#if BLUR
image blur_image(image im) {
  int x,y,k;
  image copy = copy_image(im);
  constrain_image(copy);
  IplImage *imcv = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
  int step = imcv->widthStep;
  for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(k= 0; k < im.c; ++k){
                imcv->imageData[y*step + x*im.c + k] = (unsigned char)(get_pixel(im,x,y,k)*255);
            }
        }
    }
  cvSmooth(imcv,imcv,CV_GAUSSIAN,49,49,0,0); 
  return ipl_to_image(imcv);

  
  // show_image_cv(copy,"test",imcv);
  //cvReleaseImage(&imcv);
}
#endif

void insert_sub_image(image a,image subim,int x_min, int y_min,int x_max, int y_max) {
    int i,j;
    if(x_min < 0) x_min = 0;
    if(x_min >= a.w) x_min = a.w-1;
    if(x_max < 0) x_max = 0;
    if(x_max >= a.w) x_max = a.w-1;

    if(y_min < 0) y_min = 0;
    if(y_min >= a.h) y_min = a.h-1;
    if(y_max < 0) y_max = 0;
    if(y_max >= a.h) y_max = a.h-1;

      for(i = 0; i <= x_max-x_min ; ++i){
      for (j = 0; j <= y_max - y_min ; ++j) {
	a.data[i+x_min + (j+y_min)*a.w + 0*a.w*a.h]= subim.data[i + j*subim.w + 0*subim.w*subim.h];
	  a.data[i+x_min + (j+y_min)*a.w + 1*a.w*a.h] = subim.data[i + j*subim.w + 1*subim.w*subim.h];
	a.data[i+x_min + (j+y_min)*a.w + 2*a.w*a.h] = subim.data[i + j*subim.w + 2*subim.w*subim.h]; 
      }
    }

      free_image(subim);
}

#if BLUR
void blur_sub_image(image a, int x_min, int y_min, int x_max, int y_max) {
  image subim = sub_image(a,x_min,y_min,x_max,y_max);
  image temp = blur_image(subim);
  insert_sub_image(a,temp,x_min,y_min,x_max,y_max);
}
#endif
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}

image **load_alphabet()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}


/**
 *  TO BE IMPLEMENTED! DO NOT USE!!
 *
 *
 * @param conf_1
 * @param conf_2
 * @return
 */
float new_confidence(float conf_1, float conf_2) {
    // TODO IMPLEMENT
    return 0.0;
}

/**
 * Returns a new object holder that contains the two passed in holders
 * NOTE: new_conf and new_name are necessary, since default values aren't permitted in C
 *
 * @param o1 first object to be contained within returned object holder
 * @param o2 second object to be contained within returned object holder
 * @param new_conf the confidence value to apply to this new box, separate calculation
 * @param new_name the name of the new object holder
 * @return a new, not smaller objectholder with the new values
 */
objectholder box_max_container(objectholder o1, objectholder o2, float new_conf, char *new_name) {
    objectholder to_ret;

    float a,b;

    float min_x = MIN(o1.x - o1.w/2., o2.x - o2.w/2.);
    float max_x = MAX(o1.x + o1.w/2., o2.x + o2.w/2.);
    float min_y = MIN(o1.y - o1.h/2., o2.y - o2.h/2.);
    float max_y = MAX(o1.y + o1.h/2., o2.y + o2.h/2.);

    to_ret.w = max_x - min_x;
    to_ret.h = max_y - min_y;
    to_ret.x = min_x + (to_ret.w / 2.);
    to_ret.y = min_y + (to_ret.h / 2.);

    to_ret.confidence = new_conf;
    to_ret.name = new_name;
    return to_ret;
}

void draw_objectholder(objectholder object,image im, float rgb[3], int width) {
  int left = (object.x - object.w/2.)*im.w;
  int right = (object.x+object.w/2.)*im.w;
  int top   = (object.y-object.h/2.)*im.h;
  int bot   = (object.y+object.h/2.)*im.h;

  if(left < 0) left = 0;
  if(right > im.w-1) right = im.w-1;
  if(top < 0) top = 0;
  if(bot > im.h-1) bot = im.h-1;

  draw_box_width(im,left,top,right,bot,width,rgb[0],rgb[1],rgb[2]);

}

void linear_reg(int needed_lines_x[], int needed_lines_y[], int count, double* m , double* b){
        double   sumx = 0.0;                      /* sum of x     */
        double   sumx2 = 0.0;                     /* sum of x**2  */
        double   sumxy = 0.0;                     /* sum of x * y */
        double   sumy = 0.0;                      /* sum of y     */
        double   sumy2 = 0.0;                     /* sum of y**2  */
        int i;
        for (i=0;i<count;i++){ 
            sumx  += needed_lines_x[i];       
            sumx2 += pow(needed_lines_x[i], 2);  
            sumxy += needed_lines_x[i] * needed_lines_y[i];
            sumy  += needed_lines_y[i];      
            sumy2 += pow(needed_lines_y[i],2); 
        } 

        double denom = (pow(sumx, 2) - count * sumx2);
        if (denom == 0) {
            // singular matrix. can't solve the problem.
	  printf("singular matrix");
	  denom += 1;

        }

        *m = (sumx*sumy - count*sumxy) / denom;
        *b = (sumy - *m *sumx) / count;
}
#if LANE_DETECTION
void processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame) {
    //vector<int>& left_x, left_y, right_x, right_y;
  int i;
    int needed_lines_x [lines->total];
    int needed_lines_y [lines->total];
    int needed_lines_xr [lines->total];
    int needed_lines_yr [lines->total];
    int countl = 0;
    double ml,bl;
    int countr = 0;
    double mr,br;

for(i = 0; i < lines->total; i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
        //printf("x: %d", line[0].x);
        double dx = line[1].x - line[0].x;
		double dy = line[1].y - line[0].y;
        double k = (dy/dx);

        //only want certain line segements within some the abs slope range [0.35, 1]
        if (-3.73 < k && k < -0.4){
		cvLine(temp_frame, line[0], line[1], CV_RGB(255, 0, 255), 3, CV_AA, 0);
        //need more work LINEAR Regression to draw lanes
        } if (0,4 < k && k < 4.73) {
	  // cvLine(temp_frame, line[0], line[1], CV_RGB(255, 0, 0), 2, CV_AA, 0);
	}
             
	if (-3.73 < k && k < -0.4){
	  printf("LEFT line segments slope: %f\n",k);
	  if(line[0].x < temp_frame->width*0.6) {
                needed_lines_x[countl] = line[0].x;
                needed_lines_x[countl + 1] = line[1].x;
                needed_lines_y[countl] = line[0].y;
                needed_lines_y[countl + 1] = line[1].y;
                countl += 2;
		    } 
    
	}
	if(0.4 < k && k < tan(M_PI/2))
	  {
            printf("RIGHT line segments slope: %f\n", k);
            printf("point x %d\n", line[0].x);
            printf("width*0.4 %d\n", temp_frame->width*0.4);
            //cvLine(temp_frame, line[0], line[1], CV_RGB(0, 255, 255), 2, CV_AA, 0);
            if(line[1].x >= temp_frame->width*0.4) {

                needed_lines_xr[countr] = line[0].x;
                needed_lines_xr[countr + 1] = line[1].x;
                needed_lines_yr[countr] = line[0].y;
                needed_lines_yr[countr + 1] = line[1].y;
                countr += 2;
		printf("%d\n",countr);
		    } 
        } 
    }
    linear_reg(needed_lines_x, needed_lines_y, countl, &ml, &bl);
    printf("left slope is: %f, left intercept is: %f\n", ml, bl);
    linear_reg(needed_lines_xr, needed_lines_yr, countr, &mr, &br);
    printf("right slope is: %f, right intercept is: %f\n" , mr, br);

    // Find 2 end points for right and left lines, used for drawing the line
	// y = m*x + b --> x = (y - b)/m
	

    double trap_height = 0.8;
    CvPoint* l1 = malloc(sizeof(CvPoint));
    l1->x = (temp_frame->height*1.5 - bl)/ml;
    l1->y = temp_frame->height*1.5;
    CvPoint* l2 = malloc(sizeof(CvPoint));
    l2->x = (temp_frame->height*(1-trap_height) - bl)/ml;
    l2->y = temp_frame->height*(1-trap_height);


    CvPoint* r1 = malloc(sizeof(CvPoint));
    r1->x = (temp_frame->height*1.5 - br)/mr;
    r1->y = temp_frame->height*1.5;
    CvPoint* r2 = malloc(sizeof(CvPoint));
    r2->x = (temp_frame->height*(1-trap_height) - br)/mr;
    r2->y = temp_frame->height*((1-trap_height));

    cvLine(temp_frame, *r1, *r2, CV_RGB(255, 0, 0), 3, CV_AA, 0);
    cvLine(temp_frame, *l1, *l2, CV_RGB(255, 0, 0), 3, CV_AA, 0);
    
    free(l1);
    free(l2);
    free(r1);
    free(r2); 

}
void crop(IplImage* src,  IplImage* dest, CvRect rect) {
    cvSetImageROI(src, rect); 
    cvCopy(src, dest, NULL); 
    cvResetImageROI(src); 
}


image lane_detection(image im){
    int x, y, k;

    //image half_im = sub_image(im,0,im.h/2,im.w,im.h);
    CvSize frame_size = cvSize(im.w, im.h);
    IplImage *frame = cvCreateImage(frame_size, IPL_DEPTH_8U, im.c);
    IplImage *edges = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    IplImage *grey = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    //IplImage *half_frame = cvCreateImage(cvSize(im.w/2,im.h/2), IPL_DEPTH_8U, 3);
    
    int step = frame->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(k= 0; k < im.c; ++k){
                frame->imageData[y*step + x*im.c + k] = (unsigned char)(get_pixel(im,x,y,k)*255);
            }
        }
    }
    cvCvtColor(frame, grey, CV_BGR2GRAY);
    cvSmooth(grey, grey, CV_GAUSSIAN, 5, 5, 0, 0);
    cvCanny(grey, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD, 3);
    //    cvShowImage("Canny", edges);
    double rho = 1;
    double theta = CV_PI/180;
    CvMemStorage* houghStorage = cvCreateMemStorage(0);
    CvSeq* lines = cvHoughLines2(edges, houghStorage, CV_HOUGH_PROBABILISTIC, 
        rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

    processLanes(lines, edges, frame);
    //cvReleaseMemStorage(houghStorage);
    return ipl_to_image(frame);
    //cvShowImage("HoughLine",frame);
}


void perform_lane_detection(image a, int x_min, int y_min, int x_max, int y_max) {
  image subim = sub_image(a,x_min,y_min,x_max,y_max);
  image temp = lane_detection(subim);
  insert_sub_image(a,temp,x_min,y_min,x_max,y_max);
}
#endif

void draw_detections(image im, int num, float thresh, box *boxes, float **probs, float **masks, char **names, image **alphabet, int classes, detections_queue_t* ** PERSON_DETECTIONS_ARRAY, detections_queue_t* ** BIKE_DETECTIONS_ARRAY, detections_queue_t* ** BIKER_DETECTIONS_ARRAY)
{
  
  //printf("Framenum : %d \n",FRAME_NUM );
 
  FRAME_NUM++;

  //fflush(stdout);

#if !(IMAGE)
    add_null_frame_nodes(PERSON_DETECTIONS_ARRAY);
    add_null_frame_nodes(BIKE_DETECTIONS_ARRAY);
    add_null_frame_nodes(BIKER_DETECTIONS_ARRAY);
#endif
  

   
#if LANE_DETECTION
    perform_lane_detection(im, 0, 0, im.w, im.h);
#endif

    //Setup for person/bike detections
    int i,j,isperson,isbicycle;
    char * classtype;
    const int MAX_ALLOWED_DETECTIONS = 100;
    objectholder *personlist = calloc(MAX_ALLOWED_DETECTIONS,sizeof(objectholder));
    objectholder *bikelist = calloc(MAX_ALLOWED_DETECTIONS,sizeof(objectholder));
    int personcount = 0;
    int bikecount = 0;
    objectholder object;
    // End of setup for person/bike detections

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        float prob;
        for(j = 0; j < classes; ++j){
            if (probs[i][j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }

                prob = probs[i][j];

		// printf("%s: %.0f%%\n", names[j], probs[i][j]*100);
            }
        }
        if(class >= 0){
            classtype = names[class];
            isperson = strncmp(classtype,"person",strlen("person"));
            isbicycle = strncmp(classtype,"bicycle",strlen("bicycle"));
            box b = boxes[i];

	    int offset = class*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];
	    rgb[0] = 0;
	    rgb[1] = 0;
	    rgb[2] = 0;

            #if ((BIKERS) || (SAVEJSON))
            if (0 == isperson) {
                assert(personcount < MAX_ALLOWED_DETECTIONS);
                object.x = b.x;
                object.y = b.y;
                object.w = b.w;
                object.h = b.h;
                object.confidence = prob;
                object.name = classtype;
		object.is_compound_object = 0;
		int person_x = get_grid_x(&object);
		int person_y = get_grid_y(&object);
		object.past = NULL;
#if !(IMAGE)
		find_past_object(0,&object,person_x,person_y,PERSON_DETECTIONS_ARRAY);
#endif
		if (object.past != NULL) {
		  draw_objectholder(*object.past,im,rgb,1);
		}
                personlist[personcount] = object;

                personcount++;
            }
            
            if (0 == isbicycle) {
                assert(bikecount < MAX_ALLOWED_DETECTIONS);
                object.x = b.x;
                object.y = b.y;
                object.w = b.w;
                object.h = b.h;
                object.confidence = prob;
                object.name = classtype;
		object.is_compound_object = 0;
		int bike_x = get_grid_x(&object);
		int bike_y = get_grid_y(&object);
		object.past = NULL;
#if !(IMAGE)
		find_past_object(0,&object,bike_x,bike_y,BIKE_DETECTIONS_ARRAY);
#endif
		if (object.past != NULL) {
		  draw_objectholder(*object.past,im,rgb,1);
		}

                bikelist[bikecount] = object;
                bikecount++;
            }
            #endif
            int width = im.h * .006;
	   

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;


            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            #if BLUR
	        if(isperson == 0) {
		        blur_sub_image(im,left,top,right,bot);
	        }
            #endif


	        image label;
            #if TRAJECTORYWARNINGS
	        if (alphabet) {
                if((class == 0) && (b.x > 0.45) && (b.x < 0.65)){
                    rgb[0] = red;
                    rgb[1] = 0;
                    rgb[2] = 0;
			        label = get_label(alphabet, "Caution", (im.h*.03)/10);
		        } else {
		            label = get_label(alphabet, labelstr, (im.h*.03)/10);
		        }
	        }
            #else
	        if(alphabet) {
		        label = get_label(alphabet, labelstr, (im.h*.03)/10);
	        }
            #endif

            if (masks){
                image mask = float_to_image(14, 14, 1, masks[i]);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }

            #if !(ONLYBIKERS)
	    if (!(isperson == 0) && !(isbicycle == 0)){ 
	    draw_box_width(im, left, top, right, bot, width, rgb[0], rgb[1], rgb[2]);
	            if (alphabet) {
	                draw_label(im, top + width, left, label, rgb);
	                free_image(label);
	            }
	    }
            #endif
	}
    }

	   
    


            

    #if BIKERS 
	  int *closest_bike_list = calloc(personcount,sizeof(int));
    //Initialize arrays to remember which people and bikes get combined so that we can draw the non-combined ones.
    
    int max_index = -1;
    //Find closest bike
    for (i = 0; i < personcount; i++){
        float max_IOU = -1.0;
	
        for (j = 0; j < bikecount; j++){
	  float IOU = box_iou(objectholder_to_box(personlist[i]),objectholder_to_box(bikelist[j]));//get_euclidean_distance(personlist[i], bikelist[j]);
            if(IOU > max_IOU) { // || (min_distance == -1.0)){
                max_IOU = IOU;
		max_index = j;//min_distance = dist;
            }   
        }
        closest_bike_list[i] = max_index;
     }
    //At this point we've found a list which contains, for each person, the closest bike 
    //to that person. This list has the same length
    //as the list of people.

    int width = im.h * .006;

    for (i = 0; i < personcount; ++i){
      
      if (max_index != -1) {
      if(test_compound_object(&personlist[i], &bikelist[closest_bike_list[i]])){ 
     
	  //remember_person_index[i] = 1;
	  //remember_bike_index[closest_bike_list[i].index] = 1;
	  personlist[i].is_compound_object = 1;
	  bikelist[closest_bike_list[i]].is_compound_object = 1;
	}
      }
    }
    //Now we know which people in the list of people are on bikes,
    //and thus also which people are NOT on bikes.
    //This is important because we want to have different drawing
    //rules if a person is on a bike. 

    for (i = 0; i < personcount; ++i) {
     if (personlist[i].is_compound_object == 1) {
  //If the person is on a bike, draw them as a biker, and push to the biker
  //linked list.
  
  
            objectholder compound_object = box_max_container(personlist[i], bikelist[closest_bike_list[i]], 1.0, "cyclist");

	    int person_x = get_grid_x(&personlist[i]);
	    int person_y = get_grid_y(&personlist[i]);

	    int bike_x = get_grid_x(&bikelist[closest_bike_list[i]]);
	    int bike_y = get_grid_y(&bikelist[closest_bike_list[i]]);

	      
	    int compound_x = get_grid_x(&compound_object);
	    int compound_y = get_grid_y(&compound_object);
	    
	    //find_past_object(0,&compound_object,compound_x,compound_y);
#if !(IMAGE)
	    push_obj(&PERSON_DETECTIONS_ARRAY[person_x][person_y]->head->detections,copy_objectholder(personlist[i]));
	    push_obj(&BIKE_DETECTIONS_ARRAY[bike_x][bike_y]->head->detections,copy_objectholder(bikelist[closest_bike_list[i]]));
	     // push_obj(&bike_linked_list,copy_objectholder(bikelist[personlist[i].index]));
	    compound_object.object1_index = PERSON_DETECTIONS_ARRAY[person_x][person_y]->head->detections->object;
	    compound_object.object2_index = BIKE_DETECTIONS_ARRAY[bike_x][bike_y]->head->detections->object;

	    push_obj(&BIKER_DETECTIONS_ARRAY[compound_x][compound_y]->head->detections,copy_objectholder(compound_object));

#endif
	    //	    push_obj(&biker_linked_list, copy_objectholder(compound_object));

	    int left  = (compound_object.x-compound_object.w/2.)*im.w;
            int right = (compound_object.x+compound_object.w/2.)*im.w;
            int top   = (compound_object.y-compound_object.h/2.)*im.h;
            int bot   = (compound_object.y+compound_object.h/2.)*im.h;
	    
	    
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;  

            int offset = 0;
            float red = get_color(2,offset,classes);
            float rgb[3];
	    rgb[0] = get_color(2,50,classes);
	    rgb[1] = get_color(1,50,classes);
	    rgb[2] = get_color(0,50,classes);
	        image label;

            #if TRAJECTORYWARNINGS
	        if (alphabet) {
                if((compound_object.x > 0.45) && (compound_object.x < 0.65)){
                    rgb[0] = red;
                    rgb[1] = 0;
                    rgb[2] = 0;
			        label = get_label(alphabet, "Caution", (im.h*.03)/10);
		        } else {
		            label = get_label(alphabet, "Cyclist", (im.h*.03)/10);
		        }
	         }
            #else
	        if(alphabet) {
		        label = get_label(alphabet, "Cyclist", (im.h*.03)/10);
	        }
            #endif
		draw_box_width(im,left,top,right,bot,width,rgb[0],rgb[1],rgb[2]);

            if (alphabet) {
              draw_label(im, top + width, left, label, rgb);
              free_image(label);
            }
     } 
        
     //#if !(ONLYBIKERS)	
 else {
   //If the person is NOT on a bike (and if by the ONLYBIKERS flag above we're allowed to draw such things)
   //draw the person as we would normally, and push them to the normal person linked list.
   int person_x = get_grid_x(&personlist[i]);
   int person_y = get_grid_y(&personlist[i]);

   //find_past_object(0,&personlist[i],person_x,person_y);

#if !(IMAGE)
   push_obj(&PERSON_DETECTIONS_ARRAY[person_x][person_y]->head->detections,copy_objectholder(personlist[i]));
#endif

	int left  = (personlist[i].x-personlist[i].w/2.)*im.w;
	int right = (personlist[i].x+personlist[i].w/2.)*im.w;
	int top   = (personlist[i].y-personlist[i].h/2.)*im.h;
	int bot   = (personlist[i].y+personlist[i].h/2.)*im.h;

	if(left < 0) left = 0;
	if(right > im.w-1) right = im.w-1;
	if(top < 0) top = 0;
	if(bot > im.h-1) bot = im.h-1;  
	int offset = 0;
	float red = get_color(2,offset,classes);
	float green = get_color(1,offset,classes);
	float blue = get_color(0,offset,classes);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	image label;

            #if TRAJECTORYWARNINGS
	        if (alphabet) {
                if((personlist[i].x > 0.45) && (personlist[i].x < 0.65)){
                    rgb[0] = red;
                    rgb[1] = 0;
                    rgb[2] = 0;
		    label = get_label(alphabet, "Caution", (im.h*.03)/10);
		        }

		else {
		            label = get_label(alphabet, "Person", (im.h*.03)/10);
		        }
	         }
            #else
	        if(alphabet) {
		        label = get_label(alphabet, "Person", (im.h*.03)/10);
	        }
            #endif
		draw_objectholder(personlist[i],im,rgb,width);

            if (alphabet) {
              draw_label(im, top + width, left, label, rgb);
              free_image(label);
	    }
 }
     //#endif
	
    }
    //#if !(ONLYBIKERS)
    for(i = 0; i < bikecount; ++i) {
      //If the bike was part of a bike/person compound object, we already drew it above,
      //so we just need to draw unpaired bicycles and push them to the list.
      if (0 == bikelist[i].is_compound_object) {
	int bike_x = get_grid_x(&bikelist[i]);
	int bike_y = get_grid_y(&bikelist[i]);


#if !(IMAGE)
	push_obj(&BIKE_DETECTIONS_ARRAY[bike_x][bike_y]->head->detections,copy_objectholder(bikelist[i]));
#endif

	int left  = (bikelist[i].x-bikelist[i].w/2.)*im.w;
	int right = (bikelist[i].x+bikelist[i].w/2.)*im.w;
	int top   = (bikelist[i].y-bikelist[i].h/2.)*im.h;
	int bot   = (bikelist[i].y+bikelist[i].h/2.)*im.h;

	if(left < 0) left = 0;
	if(right > im.w-1) right = im.w-1;
	if(top < 0) top = 0;
	if(bot > im.h-1) bot = im.h-1;  
	int offset = 0;
	float red = get_color(2,offset,classes);
	float rgb[3];
	rgb[0] = 0;
	rgb[1] = red;
	rgb[2] = 0;
	int width = im.h *.006;
	if (alphabet) {
	  image label = get_label(alphabet, "Bicycle", (im.h*.03)/10);
	  draw_label(im, top + width, left, label, rgb);
	  free_image(label);
	}

	draw_objectholder(bikelist[i],im,rgb,width);
      }
    }
    //#endif        

#if SAVEJSON
    int personlength = 0;
    int bikelength = 0;
    for (i = 0; i < personcount; ++i) {
      if (personlist[i].is_compound_object == 0) ++personlength;
    }
    for (i = 0; i < bikecount; ++i) {
      if (bikelist[i].is_compound_object) ++bikelength;
    }

        FILE * fp;
	fp = fopen("detectionfile_negatives.json","w+");
	writedetectionlist_non_compound(personlist,bikelist,closest_bike_list,personcount,bikecount,fp,"person and bikes");
	fclose(fp);
	
	fp = fopen("detectionfile_positives.json","w+");
	writedetectionlist_compound(personlist,bikelist,closest_bike_list,personcount,fp,"cyclist");
	fclose(fp);
#endif    

     
    free(closest_bike_list);
    free(personlist);
    free(bikelist);
    
    #endif
}


void transpose_image(image im)
{
    assert(im.w == im.h);
    int n, m;
    int c;
    for(c = 0; c < im.c; ++c){
        for(n = 0; n < im.w-1; ++n){
            for(m = n + 1; m < im.w; ++m){
                float swap = im.data[m + im.w*(n + im.h*c)];
                im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
                im.data[n + im.w*(m + im.h*c)] = swap;
            }
        }
    }
}

void rotate_image_cw(image im, int times)
{
    assert(im.w == im.h);
    times = (times + 400) % 4;
    int i, x, y, c;
    int n = im.w;
    for(i = 0; i < times; ++i){
        for(c = 0; c < im.c; ++c){
            for(x = 0; x < n/2; ++x){
                for(y = 0; y < (n-1)/2 + 1; ++y){
                    float temp = im.data[y + im.w*(x + im.h*c)];
                    im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
                    im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
                    im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
                    im.data[x + im.w*(n-1-y + im.h*c)] = temp;
                }
            }
        }
    }
}

void flip_image(image a)
{
    int i,j,k;
    for(k = 0; k < a.c; ++k){
        for(i = 0; i < a.h; ++i){
            for(j = 0; j < a.w/2; ++j){
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.w, a.h, 1);
    for(i = 0; i < a.c; ++i){
        for(j = 0; j < a.h*a.w; ++j){
            dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
        }
    }
    for(j = 0; j < a.h*a.w; ++j){
        dist.data[j] = sqrt(dist.data[j]);
    }
    return dist;
}

void ghost_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    float max_dist = sqrt((-source.w/2. + .5)*(-source.w/2. + .5));
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float dist = sqrt((x - source.w/2. + .5)*(x - source.w/2. + .5) + (y - source.h/2. + .5)*(y - source.h/2. + .5));
                float alpha = (1 - dist/max_dist);
                if(alpha < 0) alpha = 0;
                float v1 = get_pixel(source, x,y,k);
                float v2 = get_pixel(dest, dx+x,dy+y,k);
                float val = alpha*v1 + (1-alpha)*v2;
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}

void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

void normalize_image2(image p)
{
    float *min = calloc(p.c, sizeof(float));
    float *max = calloc(p.c, sizeof(float));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.h*p.w; ++i){
            float v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i){
        if(max[i] - min[i] < .000000001){
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.w*p.h; ++i){
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
    free(min);
    free(max);
}

void copy_image_into(image src, image dest)
{
    memcpy(dest.data, src.data, src.h*src.w*src.c*sizeof(float));
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

#ifdef OPENCV
void show_image_cv(image p, const char *name, IplImage *disp)
{
    int x,y,k;
    if(p.c == 3) rgbgr_image(p);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    int step = disp->widthStep;
    #if IMAGE || LIVEVIDEO
    cvNamedWindow(buff, CV_WINDOW_NORMAL); 
    #endif
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    ++windows;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(p,x,y,k)*255);
            }
        }
    }
    if(0){
        int w = 448;
        int h = w*p.h/p.w;
        if(h > 1000){
            h = 1000;
            w = h*p.w/p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_LINEAR);
        cvReleaseImage(&buffer);
    }
#if IMAGE || LIVEVIDEO
    // cvSmooth(disp,disp,CV_GAUSSIAN,11,11,0,0);
    cvShowImage(buff, disp);
#endif
#if SAVEVIDEO
    {
      CvSize size;
      size.width = disp->width;
      size.height = disp->height;

      static CvVideoWriter* output_video = NULL;    // cv::VideoWriter output_video;
      if (output_video == NULL)
	{
	  printf("\n SRC output_video = %p \n", output_video);
	  const char* output_name = "test_dnn_out.avi";
	  output_video = cvCreateVideoWriter(output_name, CV_FOURCC('D', 'I', 'V', 'X'), 25, size, 1);
	  printf("\n cvCreateVideoWriter, DST output_video = %p  \n", output_video);
	}
      cvWriteFrame(output_video, disp);
    } 
#endif
}
#endif



void show_image(image p, const char *name)
{
#ifdef OPENCV
  IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
  image copy = copy_image(p);
  constrain_image(copy);
  //blur_image(disp);
  #if IMAGE
  show_image_cv(copy, name, disp);
  #endif
  free_image(copy);
  cvReleaseImage(&disp);

#endif
#if SAVEIMAGE
  //fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image(p, name);
#endif
}

#ifdef OPENCV

void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

image load_image_cv(char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if( (src = cvLoadImage(filename, flag)) == 0 )
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    rgbgr_image(out);
    return out;
}

void flush_stream_buffer(CvCapture *cap, int n)
{
    int i;
    for(i = 0; i < n; ++i) {
        cvQueryFrame(cap);
    }
}

image get_image_from_stream(CvCapture *cap)
{
    IplImage* src = cvQueryFrame(cap);
    if (!src) return make_empty_image(0,0,0);
    image im = ipl_to_image(src);
    rgbgr_image(im);
    return im;
}

int fill_image_from_stream(CvCapture *cap, image im)
{
    IplImage* src = cvQueryFrame(cap);
    if (!src) return 0;
    ipl_into_image(src, im);
    rgbgr_image(im);
    return 1;
}

void save_image_jpg(image p, const char *name)
{
    image copy = copy_image(p);
    if(p.c == 3) rgbgr_image(copy);
    int x,y,k;

    char buff[256];
    sprintf(buff, "%s.jpg", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    cvSaveImage(buff, disp,0);
    cvReleaseImage(&disp);
    free_image(copy);
}
#endif

void save_image_png(image im, const char *name)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
#ifdef OPENCV
    save_image_jpg(im, name);
#else
    save_image_png(im, name);
#endif
}


void show_image_layers(image p, char *name)
{
    int i;
    char buff[256];
    for(i = 0; i < p.c; ++i){
        sprintf(buff, "%s - Layer %d", name, i);
        image layer = get_image_layer(p, i);
        show_image(layer, buff);
        free_image(layer);
    }
}

void show_image_collapsed(image p, char *name)
{
    image c = collapse_image_layers(p, 1);
    show_image(c, name);
    free_image(c);
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

image make_random_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    int i;
    for(i = 0; i < w*h*c; ++i){
        out.data[i] = (rand_normal() * .25) + .5;
    }
    return out;
}

image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}

void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
    int x, y, c;
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                int rx = ((float)x / w) * im.w;
                int ry = ((float)y / h) * im.h;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(canvas, x + dx, y + dy, c, val);
            }
        }
    }
}

image center_crop_image(image im, int w, int h)
{
    int m = (im.w < im.h) ? im.w : im.h;   
    image c = crop_image(im, (im.w - m) / 2, (im.h - m)/2, m, m);
    image r = resize_image(c, w, h);
    free_image(c);
    return r;
}

image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(w, h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

image rotate_image(image im, float rad)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(im.w, im.h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void translate_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
    int i;
    int best = 0;
    float best_distance = FLT_MAX;
    for(i = min; i <= max; i += 2){
        image c = crop_image(b, 0, i, b.w, b.h);
        float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
        if(d < best_distance){
            best_distance = d;
            best = i;
        }
        printf("%d %f\n", i, d);
        free_image(c);
    }
    return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
    if(!out) out = "out";
    image a = load_image(f1, 0,0,0);
    image b = load_image(f2, 0,0,0);
    int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

    image c1 = crop_image(b, 10, shift, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
    image c2 = crop_image(b, -10, shift, b.w, b.h);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

    if(d2 < d1 && 0){
        image swap = a;
        a = b;
        b = swap;
        shift = -shift;
        printf("swapped, %d\n", shift);
    }
    else{
        printf("%d\n", shift);
    }

    image c = crop_image(b, delta, shift, a.w, a.h);
    int i;
    for(i = 0; i < c.w*c.h; ++i){
        c.data[i] = a.data[i];
    }
#ifdef OPENCV
    save_image_jpg(c, out);
#else
    save_image(c, out);
#endif
}

void letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    free_image(resized);
}

image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    free_image(resized);
    return boxed;
}

image resize_max(image im, int max)
{
    int w = im.w;
    int h = im.h;
    if(w > h){
        h = (h * max) / w;
        w = max;
    } else {
        w = (w * max) / h;
        h = max;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image resize_min(image im, int min)
{
    int w = im.w;
    int h = im.h;
    if(w < h){
        h = (h * min) / w;
        w = min;
    } else {
        w = (w * min) / h;
        h = min;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image random_crop_image(image im, int w, int h)
{
    int dx = rand_int(0, im.w - w);
    int dy = rand_int(0, im.h - h);
    image crop = crop_image(im, dx, dy, w, h);
    return crop;
}

augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = {0};
    aspect = rand_scale(aspect);
    int r = rand_int(low, high);
    int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
    float scale = (float)r / min;

    float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

    float dx = (im.w*scale/aspect - w) / 2.;
    float dy = (im.h*scale - w) / 2.;
    //if(dx < 0) dx = 0;
    //if(dy < 0) dy = 0;
    dx = rand_uniform(-dx, dx);
    dy = rand_uniform(-dy, dy);

    a.rad = rad;
    a.scale = scale;
    a.w = w;
    a.h = h;
    a.dx = dx;
    a.dy = dy;
    a.aspect = aspect;
    return a;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = random_augment_args(im, angle, aspect, low, high, w, h);
    image crop = rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
    return crop;
}

float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void yuv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            y = get_pixel(im, i , j, 0);
            u = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);

            r = y + 1.13983*v;
            g = y + -.39465*u + -.58060*v;
            b = y + 2.03211*u;

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void rgb_to_yuv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);

            y = .299*r + .587*g + .114*b;
            u = -.14713*r + -.28886*g + .436*b;
            v = .615*r + -.51499*g + -.10001*b;

            set_pixel(im, i, j, 0, y);
            set_pixel(im, i, j, 1, u);
            set_pixel(im, i, j, 2, v);
        }
    }
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);
            float max = three_way_max(r,g,b);
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0){
                s = 0;
                h = 0;
            }else{
                s = delta/max;
                if(r == max){
                    h = (g - b) / delta;
                } else if (g == max) {
                    h = 2 + (b - r) / delta;
                } else {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h/6.;
            }
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            h = 6 * get_pixel(im, i , j, 0);
            s = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);
            if (s == 0) {
                r = g = b = v;
            } else {
                int index = floor(h);
                f = h - index;
                p = v*(1-s);
                q = v*(1-s*f);
                t = v*(1-s*(1-f));
                if(index == 0){
                    r = v; g = t; b = p;
                } else if(index == 1){
                    r = q; g = v; b = p;
                } else if(index == 2){
                    r = p; g = v; b = t;
                } else if(index == 3){
                    r = p; g = q; b = v;
                } else if(index == 4){
                    r = t; g = p; b = v;
                } else {
                    r = v; g = p; b = q;
                }
            }
            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void grayscale_image_3c(image im)
{
    assert(im.c == 3);
    int i, j, k;
    float scale[] = {0.299, 0.587, 0.114};
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float val = 0;
            for(k = 0; k < 3; ++k){
                val += scale[k]*get_pixel(im, i, j, k);
            }
            im.data[0*im.h*im.w + im.w*j + i] = val;
            im.data[1*im.h*im.w + im.w*j + i] = val;
            im.data[2*im.h*im.w + im.w*j + i] = val;
        }
    }
}

image grayscale_image(image im)
{
    assert(im.c == 3);
    int i, j, k;
    image gray = make_image(im.w, im.h, 1);
    float scale[] = {0.299, 0.587, 0.114};
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
            }
        }
    }
    return gray;
}

image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i){
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}

image blend_image(image fore, image back, float alpha)
{
    assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
    image blend = make_image(fore.w, fore.h, fore.c);
    int i, j, k;
    for(k = 0; k < fore.c; ++k){
        for(j = 0; j < fore.h; ++j){
            for(i = 0; i < fore.w; ++i){
                float val = alpha * get_pixel(fore, i, j, k) + 
                    (1 - alpha)* get_pixel(back, i, j, k);
                set_pixel(blend, i, j, k, val);
            }
        }
    }
    return blend;
}

void scale_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix*v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

void translate_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix+v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

image binarize_image(image im)
{
    image c = copy_image(im);
    int i;
    for(i = 0; i < im.w * im.h * im.c; ++i){
        if(c.data[i] > .5) c.data[i] = 1;
        else c.data[i] = 0;
    }
    return c;
}

void saturate_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}

void hue_image(image im, float hue)
{
    rgb_to_hsv(im);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void exposure_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 2, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}

void distort_image(image im, float hue, float sat, float val)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void random_distort_image(image im, float hue, float saturation, float exposure)
{
    float dhue = rand_uniform(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
    distort_image(im, dhue, dsat, dexp);
}

void saturate_exposure_image(image im, float sat, float exposure)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im);
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


void test_resize(char *filename)
{
    image im = load_image(filename, 0,0, 3);
    float mag = mag_array(im.data, im.w*im.h*im.c);
    printf("L2 Norm: %f\n", mag);
    image gray = grayscale_image(im);

    image c1 = copy_image(im);
    image c2 = copy_image(im);
    image c3 = copy_image(im);
    image c4 = copy_image(im);
    distort_image(c1, .1, 1.5, 1.5);
    distort_image(c2, -.1, .66666, .66666);
    distort_image(c3, .1, 1.5, .66666);
    distort_image(c4, .1, .66666, 1.5);


    show_image(im,   "Original");
    show_image(gray, "Gray");
    show_image(c1, "C1");
    show_image(c2, "C2");
    show_image(c3, "C3");
    show_image(c4, "C4");
#ifdef OPENCV
    while(1){
        image aug = random_augment_image(im, 0, .75, 320, 448, 320, 320);
        show_image(aug, "aug");
        free_image(aug);


        float exposure = 1.15;
        float saturation = 1.15;
        float hue = .05;

        image c = copy_image(im);

        float dexp = rand_scale(exposure);
        float dsat = rand_scale(saturation);
        float dhue = rand_uniform(-hue, hue);

        distort_image(c, dhue, dsat, dexp);
        show_image(c, "rand");
        printf("%f %f %f\n", dhue, dsat, dexp);
        free_image(c);
        cvWaitKey(0);
    }
#endif
}


image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}

image load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    image out = load_image_cv(filename, c);
#else
    image out = load_image_stb(filename, c);
#endif

    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

image load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}

image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}
void print_image(image m)
{
    int i, j, k;
    for(i =0 ; i < m.c; ++i){
        for(j =0 ; j < m.h; ++j){
            for(k = 0; k < m.w; ++k){
                printf("%.2lf, ", m.data[i*m.h*m.w + j*m.w + k]);
                if(k > 30) break;
            }
            printf("\n");
            if(j > 30) break;
        }
        printf("\n");
    }
    printf("\n");
}

image collapse_images_vert(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        w = (w+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, 0, h_offset);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

image collapse_images_horz(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = ims[0].h;
    h = size;
    w = (ims[0].w + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int w_offset = i*(size+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, w_offset, 0);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int h_offset = j*(size+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

void show_image_normalized(image im, const char *name)
{
    image c = copy_image(im);
    normalize_image(c);
    show_image(c, name);
    free_image(c);
}

void show_images(image *ims, int n, char *window)
{
    image m = collapse_images_vert(ims, n);
    /*
       int w = 448;
       int h = ((float)m.h/m.w) * 448;
       if(h > 896){
       h = 896;
       w = ((float)m.w/m.h) * 896;
       }
       image sized = resize_image(m, w, h);
     */
    normalize_image(m);
    save_image(m, window);
    show_image(m, window);
    free_image(m);
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
