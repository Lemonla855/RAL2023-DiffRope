#include "../default.inc"

camera{
    location <0,200,3>
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
    b_spline 30
    ,<0.0,0.0,-7.5>,0.19584950804710388
    ,<-0.015934981405735016,0.51198410987854,-7.426947116851807>,0.18720659613609314
    ,<-0.0318559929728508,1.023496150970459,-7.353828430175781>,0.1879783719778061
    ,<-0.04775126650929451,1.5341222286224365,-7.280538558959961>,0.1879875808954239
    ,<-0.06361521035432816,2.0436246395111084,-7.20690393447876>,0.18807768821716309
    ,<-0.07945601642131805,2.552185535430908,-7.132704257965088>,0.18814697861671448
    ,<-0.09530807286500931,3.0608348846435547,-7.05783224105835>,0.1881421059370041
    ,<-0.1112469807267189,3.57201886177063,-6.982735633850098>,0.18796055018901825
    ,<-0.12739874422550201,4.090056896209717,-6.9091949462890625>,0.1874634474515915
    ,<-0.143926203250885,4.620913028717041,-6.841110706329346>,0.18652181327342987
    ,<-0.16097356379032135,5.170510768890381,-6.784356117248535>,0.18512322008609772
    ,<-0.1785614788532257,5.741182804107666,-6.744577407836914>,0.1835297793149948
    ,<-0.19645535945892334,6.326916694641113,-6.7225189208984375>,0.1824115514755249
    ,<-0.21406304836273193,6.909344673156738,-6.707539081573486>,0.18279500305652618
    ,<-0.23042799532413483,7.456960678100586,-6.670015335083008>,0.18561840057373047
    ,<-0.2442757934331894,7.926619529724121,-6.550801753997803>,0.19018468260765076
    ,<-0.2534792721271515,8.246402740478516,-6.249581813812256>,0.19122394919395447
    ,<-0.2548184394836426,8.314926147460938,-5.718301296234131>,0.17803385853767395
    ,<-0.24390356242656708,7.947281360626221,-4.954196453094482>,0.16115817427635193
    ,<-0.2275136113166809,7.3962225914001465,-4.042844772338867>,0.15396635234355927
    ,<-0.20709723234176636,6.7131266593933105,-3.013324499130249>,0.14962513744831085
    ,<-0.18415598571300507,5.950881481170654,-1.9059072732925415>,0.14782322943210602
    ,<-0.16005095839500427,5.156585693359375,-0.7605308890342712>,0.14752109348773956
    ,<-0.13579316437244415,4.364136695861816,0.3915996253490448>,0.1480882316827774
    ,<-0.1119595542550087,3.5916128158569336,1.5316855907440186>,0.14899718761444092
    ,<-0.08874627202749252,2.8437070846557617,2.6525192260742188>,0.14988663792610168
    ,<-0.06610221415758133,2.1168975830078125,3.754906177520752>,0.1504877209663391
    ,<-0.04386938735842705,1.4045960903167725,4.843498706817627>,0.15129493176937103
    ,<-0.021880540996789932,0.7005336880683899,5.923661708831787>,0.14863796532154083
    ,<0.0,0.0,7.0>,0.1689179241657257
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
