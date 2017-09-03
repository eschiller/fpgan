# FPGAN

## Dependencies

 * xmltodict
 * tensorflow
 * numpy
 * json
 

## Schedule
 
DATE | TASK
--- | ---
8/23 | Obtain a set of 5 floor plans and create .svg representations. 
9/6 | Create preprocessing tools for dataset and apply tool transformations. This will result in a set of .svg files with normalized coordinates, with all max y-coordinate of 32 and all points snapped to integer values.
9/13 | Develop tool to convert dataset from .svg files to JSON format.
9/20 | Develop tool to convert JSON format data to NumPy matrices, resulting in 5 8x8x5 or 16x16x5 integer matrices.
10/11 | Build initial GAN in Tensorflow and run training on single floor plan. The deliverable at this point will be a GAN that essentially models the identity function, demonstrating that basic learning is functioning as expected.
3 weeks | Obtain remaining dataset floor plans (~100) and create .svg representations.
1 week | Run GAN on full dataset with expectation of generating novel floor plans. Given that the overfitting reduction won’t have been implemented yet, the generated floor plans will likely be overfitted to be the same image, but this should give a baseline from which to work from.
2 weeks | Write, in python, data transformation tools to reduce overfitting and integrate these with GAN training. Ideally, this will result in improvements in generated data, most notably in the variety of floor plans which are generated.
1 weeks | Rerun training while tuning hyperparameters to optimize performance, resulting in decreased training speeds.
2 weeks | Tune hyperparameters for improved results, with intended deliverable of more believable generated floor plans
1 week | Revisit any suggested optimizations from literature (or any new developments) and apply to GAN as needed, with goal of improved results.
