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
    ,<0.07096522075165693,-0.0010107688448346137,2.288704248742252>,0.05
    ,<0.025924770713810368,-0.0010555749717396496,2.3104162258367436>,0.05
    ,<-0.019377531436097602,-0.00110215251616337,2.3315870083952532>,0.05
    ,<-0.06411013109882113,-0.0011396834478795087,2.3539455359942614>,0.05
    ,<-0.10670035570520495,-0.0011478620765475423,2.380158403703201>,0.05
    ,<-0.14479176675659935,-0.001103860737985206,2.412559418768569>,0.05
    ,<-0.17518267558560244,-0.0009913739847062526,2.452265125307412>,0.05
    ,<-0.19401057414747547,-0.0008110838515883947,2.4985754797356376>,0.05
    ,<-0.19741287367791685,-0.0005893502342503765,2.548440448705826>,0.05
    ,<-0.18337501606573112,-0.000369932481502889,2.5964038719859586>,0.05
    ,<-0.15389391629608276,-0.0001947173210580879,2.6367606761372047>,0.05
    ,<-0.11437647807481394,-8.929497364676455e-05,2.6673712678211183>,0.05
    ,<-0.07056965887032914,-5.646707349670356e-05,2.691463027832859>,0.05
    ,<-0.02694752333143875,-8.125289819843837e-05,2.715892693186986>,0.05
    ,<0.011774163693726192,-0.00014099960133109676,2.7475129213771785>,0.05
    ,<0.03943429473410932,-0.00021484283672872727,2.7891460191745896>,0.05
    ,<0.05099559866787909,-0.00029094593081659766,2.837767911297179>,0.05
    ,<0.04622359739603794,-0.00036925251489995094,2.887515389699531>,0.05
    ,<0.029495991498759606,-0.0004567350379716797,2.934611761570041>,0.05
    ,<0.0065289323542650456,-0.0005550583132655788,2.97900768541151>,0.05
    ,<-0.01826331900081417,-0.0006580446494148858,3.022417019422994>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
