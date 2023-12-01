\ Model smartCurb
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  OBJ0: Priority=1 Weight=100 AbsTol=1e-06 RelTol=0
   - 106.5625 x_i_j[1,0] - 106.5625 x_i_j[1,1] - 106.5625 x_i_j[1,2]
   - 106.5625 x_i_j[1,3] - 12 x_i_j[2,0] - 12 x_i_j[2,1] - 12 x_i_j[2,2]
   - 12 x_i_j[2,3] - 10 x_i_j[3,0] - 10 x_i_j[3,1] - 10 x_i_j[3,2]
   - 10 x_i_j[3,3] + 128.5625 Constant0
  OBJ1: Priority=1 Weight=0 AbsTol=1e-06 RelTol=0
   - 1.0625 x_i_j[1,0] - 1.0625 x_i_j[1,1] - 1.0625 x_i_j[1,2]
   - 1.0625 x_i_j[1,3] + 1.0625 Constant1
  OBJ2: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ6: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ7: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ8: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ9: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ10: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ11: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ12: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ13: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ14: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ15: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ16: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ17: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ18: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ19: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ20: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ21: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ22: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ23: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ24: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ25: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ26: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ27: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ28: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ29: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ30: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ31: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ32: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ33: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ34: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ35: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ36: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ37: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ38: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ39: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ40: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ41: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ42: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ43: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ44: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ45: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ46: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ47: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ48: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  OBJ49: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
  
  AbsDev: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
   t_i[0] + t_i[1] + t_i[2] - 2457 Constant50
Subject To
 diag-0: x_i_j[0,0] = 0
 diag-1: x_i_j[1,1] = 0
 diag-2: x_i_j[2,2] = 0
 diag-3: x_i_j[3,3] = 0
 Prev-T-4: t_i[0] = 745
 Prev-T-8: t_i[2] = 853
 Prev-X-1: x_i_j[1,0] + x_i_j[1,1] + x_i_j[1,2] + x_i_j[1,3] >= 1
 Prev-X-3: x_i_j[3,0] + x_i_j[3,1] + x_i_j[3,2] + x_i_j[3,3] >= 0
 NumSpaces: x_i_j[0,0] + x_i_j[0,1] + x_i_j[0,2] + x_i_j[0,3] <= 1
 FlowPres-0: x_i_j[0,1] + x_i_j[0,2] + x_i_j[0,3] - x_i_j[1,0] - x_i_j[2,0]
   - x_i_j[3,0] = 0
 FlowPres-1: - x_i_j[0,1] + x_i_j[1,0] + x_i_j[1,2] + x_i_j[1,3]
   - x_i_j[2,1] - x_i_j[3,1] = 0
 FlowPres-2: - x_i_j[0,2] - x_i_j[1,2] + x_i_j[2,0] + x_i_j[2,1]
   + x_i_j[2,3] - x_i_j[3,2] = 0
 FlowPres-3: - x_i_j[0,3] - x_i_j[1,3] - x_i_j[2,3] + x_i_j[3,0]
   + x_i_j[3,1] + x_i_j[3,2] = 0
 SingleAssignment-1: x_i_j[1,0] + x_i_j[1,1] + x_i_j[1,2] + x_i_j[1,3]
   <= 1
 SingleAssignment-2: x_i_j[2,0] + x_i_j[2,1] + x_i_j[2,2] + x_i_j[2,3]
   <= 1
 SingleAssignment-3: x_i_j[3,0] + x_i_j[3,1] + x_i_j[3,2] + x_i_j[3,3]
   <= 1
 TimeOverlap;_0-1: - t_i[0] + t_i[1] - 67 x_i_j[1,2] >= 98
 TimeOverlap;_0-2: - t_i[0] + t_i[2] - 61 x_i_j[1,3] >= 104
 TimeOverlap;_1-0: t_i[0] - t_i[1] - 130 x_i_j[2,1] >= -118
 TimeOverlap;_1-2: - t_i[1] + t_i[2] - 16 x_i_j[2,3] >= -4
 TimeOverlap;_2-0: t_i[0] - t_i[2] - 134 x_i_j[3,1] >= -124
 TimeOverlap;_2-1: t_i[1] - t_i[2] - 26 x_i_j[3,2] >= -16
 ParkPast-0: t_i[0] <= 1275
 Start-0: t_i[0] >= 0
 End-0: t_i[0] <= 1440
 ParkPast-1: t_i[1] <= 1428
 Start-1: t_i[1] >= 0
 End-1: t_i[1] <= 1440
 ParkPast-2: t_i[2] <= 1430
 Start-2: t_i[2] >= 0
 End-2: t_i[2] <= 1440
 StartShift_-_0: t_i[0] >= 745
 EndShift_-_0: t_i[0] - 10 x_i_j[1,0] - 10 x_i_j[1,1] - 10 x_i_j[1,2]
   - 10 x_i_j[1,3] <= 745
 StartShift_-_1: t_i[1] >= 853
 EndShift_-_1: t_i[1] - 10 x_i_j[2,0] - 10 x_i_j[2,1] - 10 x_i_j[2,2]
   - 10 x_i_j[2,3] <= 853
 StartShift_-_2: t_i[2] >= 859
 EndShift_-_2: t_i[2] - 10 x_i_j[3,0] - 10 x_i_j[3,1] - 10 x_i_j[3,2]
   - 10 x_i_j[3,3] <= 859
Bounds
 745 <= t_i[0] <= 755
 853 <= t_i[1] <= 863
 859 <= t_i[2] <= 869
 Constant0 = 1
 Constant1 = 1
 Constant50 = 1
Binaries
 x_i_j[0,0] x_i_j[0,1] x_i_j[0,2] x_i_j[0,3] x_i_j[1,0] x_i_j[1,1]
 x_i_j[1,2] x_i_j[1,3] x_i_j[2,0] x_i_j[2,1] x_i_j[2,2] x_i_j[2,3]
 x_i_j[3,0] x_i_j[3,1] x_i_j[3,2] x_i_j[3,3]
End
