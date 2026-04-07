The GraSP benchmark provides multi-granular annotations for four tasks in Radical Prostatectomy videos: surgical phase recognition, surgical steps recognition, surgical instrument segmentation, and surgical atomic action detection. All annotations are provided in COCO JSON-like format with the following naming convention:

grasp_<task-type>_<split>.json

***TASK TYPES AND ANNOTATIONS***
We categorize the tasks into two types: long-term and short-term.


Long-term tasks include surgical phase recognition and surgical step recognition. As we base our format in the COCO annotation format, annotation files named with "long-term " type contain:

    - Lists of phase and step labels.
    - List of selected keyframes at 1-second intervals.
    - Each keyframe's phase and step category (there is only one phase and step label per frame, so the list of annotations has the same length as the list of keyframes).

{
  "phases_categories": [
    			{"id": <phase category id (0-10)>, 
			"name": <phase category name>, 
			"description": <surgical description of phase category>,
			"supercategory": <phase category type>}, 
			...],

  "steps_categories": [
    			{"id": <step category id (0-20)>, 
			 "name": <step category name>, 
			 "description": <surgical description of step category>,
			 "supercategory": <step category type>}, 
			 ...],

  "images": [
    	     {"id": <keyframe id>, 
	      "file_name": <CASE_name/frame_identifier.jpg>, 
	      "width": 1280, 
	      "height": 800, 
	      "video_name": <CASE name>, 
	      "frame_num": <frame number>}, 
	      ...],

  "annotations": [
   		 {"id": <annotation id>, 
		  "image_id": <id of corresponding frame>, 
		  "image_name": <CASE_name/frame_identifier.jpg>, 
		  "phases": <phase category id>, 
		  "steps": <step category id>}, 
		  ...]
}



Short-term tasks correspond to surgical instrument segmentation and surgical atomic visual action detection. Similar to the COCO format, these files contain:
    - List of surgical instruments, atomic actions, phases, and steps label names.
    - List of selected keyframes at 35s intervals.
    - List of instrument instances with their segmentation mask, instrument type category, list of atomic actions performed, and the phase and step category of its corresponding frame.

{
  "categories": [
    		{"id": <instrument category id (1-7)>, 
		"name": <instrument category name>, 
		"supercategory": <instrument category type>}, 
		...],

  "actions_categories": [
    			 {"id": <action category id (1-14)>, 
			  "name": <action category name>, 
			  "supercategory": <action category type>}, 
			  ...],

  "phases_categories": [
    			{"id": <phase category id (0-10)>, 
			"name": <phase category name>, 
			"description": <surgical description of phase category>,
			"supercategory": <phase category type>}, 
			...],

  "steps_categories": [
    			{"id": <step category id (0-20)>, 
			 "name": <step category name>, 
			 "description": <surgical description of step category>,
			 "supercategory": <step category type>}, 
			 ...],

  "images": [
    	     {"id": <keyframe id>, 
	      "file_name": <CASE_name/frame_identifier.jpg>, 
	      "width": 1280, 
	      "height": 800, 
              "video_name": <CASE name>, 
              "frame_num": <frame number>}, 
	      ...],

  "annotations": [
    		  {"id": <annotation id>, 
		   "image_id": <id of corresponding frame>, 
		   "image_name": <CASE_name/frame_identifier.jpg>, 
		   "segmentation": <instrument mask in RLE format>, 
		   "bbox": <instrument bounding box in x,y,w,h format (corresponding to segmentation)>, 
		   "area": <area of segmentation mask>, 
		   "iscrowd": 0, 
		   "category_id": <instrument category id>, 
		   "actions": [<action category id>, 
			       <action category id>, 
			       ...], 
		   "phases": <phase category id of corresponding frame>, 
		   "steps": <step category id of corresponding frame>}, 
		   ...]
}

***ADDITIONAL DETAILS***
- We also provide the semantic segmentation masks for each annotated frame stored as one-channel PNG images in the "segmentation" directory. Each pixel value in these images corresponds to an instrument-type label or the background. The pixel values in these PNG images range from 0 (representing the background) to the maximum instrument label, which is 7.
- The training set includes CASEs numbered 1 to 21. This set is further divided into two folds for cross-validation purposes. The testing set comprises CASEs numbered 41 to 53. These cases are reserved exclusively for evaluating the performance of models trained on the training set. Each annotation file is named according to its corresponding split (train or test) or fold (fold1 or fold2).

***IMPORTANT NOTES***
-Long-term annotations include frames at 1-second intervals, while short-term annotations include frames at 35-second intervals due to resource constraints.
- Each instrument has at least one atomic action, with a maximum of three atomic action labels.
- Phase and step labels start from 0 (Idle), while instrument and action categories begin from 1.
- The segmentation masks are in PNG format with values ranging from 0 to 255. Due to the limited range of instrument labels, the masks may appear dark when viewed directly as PNG images. For proper visualization, it is recommended that these masks be opened and processed programmatically.