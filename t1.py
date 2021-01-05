import Main_Execute as mexe


results_dict = mexe.run_optimizer(problem_testbench="DTLZ", 
                                                                    problem_name="DTLZ2", 
                                                                    nobjs=2, 
                                                                    nvars=10, 
                                                                    sampling="LHS", 
                                                                    folder_data="AM_Samples_109_Final",
                                                                    is_data=True,
                                                                    run=0,
                                                                    approach=82)