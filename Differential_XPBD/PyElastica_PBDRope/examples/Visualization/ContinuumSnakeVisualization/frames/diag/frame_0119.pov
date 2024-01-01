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
    ,<0.04670995186906979,-0.0007192025555653672,2.8337579633799628>,0.05
    ,<0.001487825360525317,-0.0008075245856889371,2.8550889100060286>,0.05
    ,<-0.04392116390941749,-0.0008962862873468055,2.8760308763087967>,0.05
    ,<-0.08864052165718732,-0.0009734598617017245,2.898417351810218>,0.05
    ,<-0.1310391479064757,-0.001017814466887181,2.924940518257161>,0.05
    ,<-0.16867587173061502,-0.0010059129401114152,2.9578698124649687>,0.05
    ,<-0.19827205612052423,-0.0009213755582878408,2.9981718068449688>,0.05
    ,<-0.21595095758075192,-0.0007662905719178514,3.0449327664782495>,0.05
    ,<-0.21798490242614918,-0.0005678405937685009,3.094872375555196>,0.05
    ,<-0.20266302869291855,-0.000369428518888027,3.1424418409190884>,0.05
    ,<-0.1722804222529616,-0.00021289811219695086,3.182125587806029>,0.05
    ,<-0.13232721656933197,-0.00012348015136253042,3.2121675310830504>,0.05
    ,<-0.08844041047744733,-0.00010370886675366994,3.2361166517748936>,0.05
    ,<-0.04503529066762383,-0.00013878187974508932,3.260932935099672>,0.05
    ,<-0.006947051242069712,-0.00020664346116067798,3.2933151578333892>,0.05
    ,<0.01955573457513902,-0.00028744255419038467,3.3356949658152764>,0.05
    ,<0.029602331251102933,-0.00037087419360346815,3.3846521435960155>,0.05
    ,<0.023310442155338926,-0.0004588738004459223,3.434229641433806>,0.05
    ,<0.005299935322512865,-0.0005587425920131495,3.480849973571944>,0.05
    ,<-0.01869110755588554,-0.00067103989217547,3.524700639038629>,0.05
    ,<-0.044360052212668415,-0.0007888958950927095,3.567596900533049>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
