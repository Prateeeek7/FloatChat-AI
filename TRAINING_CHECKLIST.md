# ARGO Model Training Checklist

## 🚀 Pre-Training Setup

### Data Preparation
- [ ] Download new ARGO NetCDF files
- [ ] Run `fix_coordinates.py` to extract real coordinates
- [ ] Verify coordinate ranges are ocean locations
- [ ] Check data distribution across ocean regions
- [ ] Validate date ranges and formats

### System Check
- [ ] Monitor available memory (target: < 80% usage)
- [ ] Check available storage space
- [ ] Verify database performance
- [ ] Test data loading speed
- [ ] Ensure MPS is available for PyTorch

## 🔧 Data Processing

### NetCDF Processing
- [ ] Use `argo_processor_corrected.py` for processing
- [ ] Handle Julian day conversion errors gracefully
- [ ] Extract real coordinates (not synthetic)
- [ ] Normalize temperature/salinity profiles
- [ ] Pad profiles to consistent length (2000m)

### Database Updates
- [ ] Update database with real coordinates
- [ ] Verify all profiles are ocean locations
- [ ] Check data quality metrics
- [ ] Optimize database indexes
- [ ] Test query performance

## 🤖 Model Training

### Architecture Selection
- [ ] LSTM for sequential data
- [ ] 1D CNN for pattern recognition
- [ ] Progressive training phases
- [ ] MPS optimization for M4 chip

### Training Process
- [ ] Start with smaller dataset (1000 profiles)
- [ ] Gradually increase data size
- [ ] Monitor memory usage and temperature
- [ ] Save model checkpoints regularly
- [ ] Track training/validation loss

### Quality Assurance
- [ ] Validate predictions on test data
- [ ] Check for overfitting
- [ ] Test model on server environment
- [ ] Create visualization of results
- [ ] Document performance metrics

## 📊 Dashboard Updates

### Visualization
- [ ] Update map with new data
- [ ] Verify ocean region boundaries
- [ ] Test interactive features
- [ ] Check performance with larger dataset
- [ ] Update data quality metrics

### Performance
- [ ] Limit map markers for smooth rendering
- [ ] Use caching for data loading
- [ ] Optimize database queries
- [ ] Test loading times
- [ ] Monitor memory usage

## ✅ Success Criteria

### Data Quality
- [ ] > 99% processing success rate
- [ ] 100% ocean coordinate accuracy
- [ ] > 95% valid profiles
- [ ] Consistent temporal coverage

### Model Performance
- [ ] > 90% training accuracy
- [ ] Realistic predictions
- [ ] Good generalization
- [ ] < 1 second inference time

### System Performance
- [ ] < 80% memory usage
- [ ] > 100 profiles/minute processing
- [ ] < 1 second database queries
- [ ] < 5 seconds dashboard loading

## 🚨 Common Pitfalls to Avoid

### Data Issues
- ❌ Don't use synthetic coordinates
- ❌ Don't ignore coordinate validation
- ❌ Don't skip Julian day error handling
- ❌ Don't assume all data is Indian Ocean

### Training Issues
- ❌ Don't start with full dataset
- ❌ Don't ignore memory monitoring
- ❌ Don't skip model checkpoints
- ❌ Don't forget validation testing

### System Issues
- ❌ Don't exceed memory limits
- ❌ Don't ignore system temperature
- ❌ Don't skip performance monitoring
- ❌ Don't forget error handling

## 📝 Documentation

### Required Documentation
- [ ] Update `ARGO_TRAINING_NOTES.md`
- [ ] Document new dataset statistics
- [ ] Record performance metrics
- [ ] Note any issues encountered
- [ ] Update success criteria

### Code Documentation
- [ ] Comment new functions
- [ ] Update README.md
- [ ] Document configuration changes
- [ ] Record model parameters
- [ ] Update deployment notes

---

**Quick Reference**: Always check `ARGO_TRAINING_NOTES.md` for detailed information before starting new training sessions.
