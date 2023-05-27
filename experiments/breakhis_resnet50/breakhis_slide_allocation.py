def allocate_slides_to_datasets():
    
    # ----- BENIGN -----------------------------------------------------------------------------------------
    training_benign_slides = ['SOB_B_A_14-22549CD', 'SOB_B_A_14-22549G',
                   'SOB_B_PT_14-21998AB',
                   'SOB_B_F_14-14134E', 'SOB_B_F_14-14134', 'SOB_B_F_14-21998EF', 'SOB_B_F_14-23060CD', 'SOB_B_F_14-21998CD', 'SOB_B_F_14-29960AB', 
                   'SOB_B_TA_14-16184CD', 'SOB_B_TA_14-16184', 'SOB_B_TA_14-19854C']

    validation_benign_slides = ['SOB_B_A_14-22549AB',
                                'SOB_B_PT_14-29315EF',
                                'SOB_B_F_14-23060AB', 'SOB_B_F_14-25197',
                                'SOB_B_TA_14-21978AB', 'SOB_B_TA_14-13200']
    
    test_benign_slides = ['SOB_B_A_14-29960CD',
                          'SOB_B_PT_14-22704',
                          'SOB_B_F_14-23222AB', 'SOB_B_F_14-9133',
                          'SOB_B_TA_14-15275', 'SOB_B_TA_14-3411F']
    
    assert len(list(set(training_benign_slides)&set(validation_benign_slides)))==0
    assert len(list(set(training_benign_slides)&set(test_benign_slides)))==0
    assert len(list(set(validation_benign_slides)&set(test_benign_slides)))==0
    print('No benign slides overlap in training, validation and test sets!')
    
    benign_slides = [training_benign_slides, validation_benign_slides, test_benign_slides]
    
    # ----- MALIGNANT --------------------------------------------------------------------------------------
    
    training_dc_slides = ['SOB_M_DC_14-13993', 'SOB_M_DC_14-14946', 'SOB_M_DC_14-15696', 'SOB_M_DC_14-2773', 
                          'SOB_M_DC_14-16188', 'SOB_M_DC_14-2985', 'SOB_M_DC_14-4364', 'SOB_M_DC_14-9461', 
                          'SOB_M_DC_14-16716', 'SOB_M_DC_14-15572', 'SOB_M_DC_14-6241', 'SOB_M_DC_14-2980', 
                          'SOB_M_DC_14-11951', 'SOB_M_DC_14-20629', 'SOB_M_DC_14-17901', 'SOB_M_DC_14-16448', 
                          'SOB_M_DC_14-16336', 'SOB_M_DC_14-20636', 'SOB_M_DC_14-18650', 'SOB_M_DC_14-14015', 
                          'SOB_M_DC_14-17614']
    training_malignant_slides = ['SOB_M_LC_14-15570C','SOB_M_LC_14-13412', 'SOB_M_LC_14-15570',
                             'SOB_M_MC_14-10147', 'SOB_M_MC_14-16456', 'SOB_M_MC_14-19979C', 'SOB_M_MC_14-19979', 'SOB_M_MC_14-13418DE', 'SOB_M_MC_14-12773',
                             'SOB_M_PC_14-12465', 'SOB_M_PC_14-9146', 'SOB_M_PC_14-19440'] + training_dc_slides
    
    validation_dc_slides = ['SOB_M_DC_14-11520', 'SOB_M_DC_14-2523', 'SOB_M_DC_14-14926', 'SOB_M_DC_14-16875', 
                            'SOB_M_DC_14-15792', 'SOB_M_DC_14-11031', 'SOB_M_DC_14-5695', 'SOB_M_DC_14-5694',
                            'SOB_M_DC_14-10926']
    validation_malignant_slides = ['SOB_M_LC_14-12204',
                                   'SOB_M_MC_14-13413',
                                   'SOB_M_PC_14-15687B', 'SOB_M_PC_15-190EF'] + validation_dc_slides
    
    test_dc_slides = ['SOB_M_DC_14-13412', 'SOB_M_DC_14-4372', 'SOB_M_DC_14-5287', 
                      'SOB_M_DC_14-8168', 'SOB_M_DC_14-12312', 'SOB_M_DC_14-3909', 'SOB_M_DC_14-16601', 
                      'SOB_M_DC_14-17915']
    test_malignant_slides = ['SOB_M_LC_14-16196',
                             'SOB_M_MC_14-18842D', 'SOB_M_MC_14-18842',
                             'SOB_M_PC_14-15704'] + test_dc_slides
    
    assert len(list(set(training_malignant_slides)&set(validation_malignant_slides)))==0
    assert len(list(set(training_malignant_slides)&set(test_malignant_slides)))==0
    assert len(list(set(validation_malignant_slides)&set(test_malignant_slides)))==0
    print('No malignant slides overlap in training, validation and test sets!')
    
    malignant_slides = [training_malignant_slides, validation_malignant_slides, test_malignant_slides]
    
    return benign_slides, malignant_slides




'''
# HELPING CODE

path = '/content/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign'
print('BENIGN')
tumor_type = os.listdir(path + '/SOB/')
for tumor in tumor_type:
  print('------',tumor,'-------')
  for slide in os.listdir(path + '/SOB/' + tumor):
      #imgs_per_size = []
      total = 0
      for imgs in os.listdir(path + '/SOB/' + tumor + '/' + slide):
          #imgs_per_size.append(len(os.listdir(path + '/SOB/' + tumor + '/' + slide + '/' + imgs)))
          total += len(os.listdir(path + '/SOB/' + tumor + '/' + slide + '/' + imgs))
      print('{} - total images:{}'.format(slide, total))
      print('\n')
      
path = '/content/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant'
print('MALIGNANT')
tumor_type = os.listdir(path + '/SOB/')
for tumor in tumor_type:
  print('------',tumor,'-------')
  for slide in os.listdir(path + '/SOB/' + tumor):
      #imgs_per_size = []
      total = 0
      for imgs in os.listdir(path + '/SOB/' + tumor + '/' + slide):
          #imgs_per_size.append(len(os.listdir(path + '/SOB/' + tumor + '/' + slide + '/' + imgs)))
          total += len(os.listdir(path + '/SOB/' + tumor + '/' + slide + '/' + imgs))
      print('{} - total images:{}'.format(slide, total))
      print('\n')

'''