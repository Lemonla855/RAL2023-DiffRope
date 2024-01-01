#include "../default.inc"

camera{
    location <15.0,10.5,-15.0>
    angle 30
    look_at <4.0,2.7,2.0>
    sky <0,1,0>
    right x*image_width/image_height
}
light_source{
    <15.0,10.5,-15.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.0201996363854951,-0.0024028021104592822,0.15013630266248854>,0.05
    ,<-0.05203264455861796,-0.002031722179825983,0.18867088163992146>,0.05
    ,<-0.079798688240599,-0.001644188164088303,0.23023286522756964>,0.05
    ,<-0.09919213387832711,-0.0012348506354186298,0.2762998171191939>,0.05
    ,<-0.10659341739127036,-0.0008198202149568385,0.3257304883889401>,0.05
    ,<-0.10015332715975196,-0.00044007266924368294,0.37529528529846184>,0.05
    ,<-0.08032307561143412,-0.00014307851564084335,0.421178972041833>,0.05
    ,<-0.04950931461853299,4.001797143040982e-05,0.46054659967304296>,0.05
    ,<-0.01120782929030764,0.00010079481712966283,0.4926893907171021>,0.05
    ,<0.030823040824614827,5.374628783934374e-05,0.5197861685508648>,0.05
    ,<0.07296027640786681,-6.892699846185319e-05,0.5467186780332237>,0.05
    ,<0.11083740878523674,-0.000225248904807288,0.5793615124258304>,0.05
    ,<0.13798899781504254,-0.0003721252668938928,0.6213349242154398>,0.05
    ,<0.14718575480622834,-0.0004762304508398222,0.6704583102255542>,0.05
    ,<0.13536992366035638,-0.0005260850617090756,0.7190107419797312>,0.05
    ,<0.10652092517406685,-0.0005359826015015884,0.7598123345649>,0.05
    ,<0.06798979173813763,-0.0005335116212463083,0.7916408203273347>,0.05
    ,<0.025768685721691154,-0.0005376687614371325,0.8183971807356613>,0.05
    ,<-0.01649516630810369,-0.0005514681334411182,0.845096326035225>,0.05
    ,<-0.05677280305321089,-0.0005704649619483923,0.8747127364971531>,0.05
    ,<-0.09486771939797786,-0.0005905313992540482,0.9070930144917101>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
