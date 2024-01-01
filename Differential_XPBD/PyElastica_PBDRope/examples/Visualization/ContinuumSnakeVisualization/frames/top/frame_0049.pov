#include "../default.inc"

camera{
    location <0,15,3>
    angle 30
    look_at <0.0,0,3>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <0.0,8.0,5.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.13490399368193748,-0.0004121517128460964,0.9161129931835035>,0.05
    ,<-0.0923860777099643,-0.0005194463287466973,0.9424242174256361>,0.05
    ,<-0.049337783209794504,-0.0006293822363236355,0.9678666492002255>,0.05
    ,<-0.0065996733024647385,-0.0007324998229146688,0.9938343866020425>,0.05
    ,<0.0341835703155853,-0.000809146441427691,1.0227797477000709>,0.05
    ,<0.07066934397264173,-0.0008354311181321079,1.056982852859756>,0.05
    ,<0.0997553581775171,-0.0007916131629222267,1.097661118692942>,0.05
    ,<0.1176690473727664,-0.000672149950271347,1.1443425122438575>,0.05
    ,<0.12048298367026956,-0.0004968150311948563,1.1942546195151942>,0.05
    ,<0.10589730474166038,-0.00030527614718299593,1.2420643622515628>,0.05
    ,<0.07559338496527465,-0.00013944502818454587,1.281814560150745>,0.05
    ,<0.034898511639176025,-2.9526872545585878e-05,1.3108449770599044>,0.05
    ,<-0.010407590116071362,1.5400978025956776e-05,1.3319821160745455>,0.05
    ,<-0.05618495556692444,5.7697244885644e-06,1.3520872348032664>,0.05
    ,<-0.09854861991420255,-3.7248968222378195e-05,1.3786391290325863>,0.05
    ,<-0.13196412784375425,-9.167905126547731e-05,1.4158235729961914>,0.05
    ,<-0.1509356686917372,-0.000143403609366781,1.4620719964708604>,0.05
    ,<-0.15393530169901942,-0.00019064045774848102,1.5119679503093444>,0.05
    ,<-0.14434198661926345,-0.0002411941459777385,1.5610250530016472>,0.05
    ,<-0.12770674355203923,-0.0002993770643504971,1.608166023440051>,0.05
    ,<-0.10880086671652496,-0.000361151495356616,1.6544472636551297>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
