
## Transformations Store
- includes objects that apply sample, or data level transformations
- we can chain these transformations together via the pipeline manager

**Example**

The following example includes a savgol transform followed by a low pass filter
```python
pipeline = [savgol_filter_transform(window_length=5, polyorder=3),
                    low_pass_fft(lowest_freq_frac=0.5, buffer_period_frac=0.2)]

pipeline = transform_pipeline(pipeline=pipeline)

```

## Preprocessing Store
### Dataframe Column Level Transforms: 
- apply to the entire split (fit on train, apply on train and test)
- typically includes column2column transformations
    - **column2transform** is used to map columns to transformation objects
- examples: standard deviation, temporal difference, log tranform etc
- TODO: include processing steps that take in multiple columns as input

### Sample Level Transforms:
- For some techniques, we would need to apply sample level transformations
- In the case of Regression
  - Each sample has a specific input and output period
  - We may need to smooth each function without knowledge of the future
      

