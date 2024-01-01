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
    ,<0.119391875647924,-0.0015222428837826385,1.2001114857500985>,0.05
    ,<0.07472590704077423,-0.0015190508595633537,1.222580805036926>,0.05
    ,<0.029644984912531574,-0.0015185196755801733,1.24421699717796>,0.05
    ,<-0.01510745270835576,-0.0015115191209149055,1.2665348180414686>,0.05
    ,<-0.05807124663649146,-0.001479134552376024,1.2921308256824808>,0.05
    ,<-0.09706316683717005,-0.0013988270890235213,1.3234427822886317>,0.05
    ,<-0.1290558259965731,-0.0012531512285644535,1.3618693970383635>,0.05
    ,<-0.15024430335208352,-0.0010400687213797488,1.407148199535217>,0.05
    ,<-0.15654175479385554,-0.0007839303100059063,1.4567292767945155>,0.05
    ,<-0.14534899247931066,-0.0005290416048725945,1.5054327019954228>,0.05
    ,<-0.11799768425884473,-0.00032017302227215524,1.5472584149057167>,0.05
    ,<-0.07963124811833627,-0.00018738247472044578,1.5792962164189621>,0.05
    ,<-0.03619569631474234,-0.00013731428274755988,1.6040497038083084>,0.05
    ,<0.007650680991348901,-0.00015693969621928902,1.6280761004083042>,0.05
    ,<0.047363421751769164,-0.00022395970990069257,1.6584440887058247>,0.05
    ,<0.07701575529838098,-0.00031669984898043764,1.6986825040407387>,0.05
    ,<0.09132312908345448,-0.0004213386200301804,1.7465667926108592>,0.05
    ,<0.0893822039164967,-0.0005350921633441539,1.796502410110396>,0.05
    ,<0.0750668599995665,-0.00066250368978852,1.8443842901774308>,0.05
    ,<0.054022732802950904,-0.0008035610134010937,1.8897207579356283>,0.05
    ,<0.03086438402475362,-0.0009507272482303692,1.9340207528613949>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
