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
    ,<-0.15071648332633886,-0.0010631531561760606,3.673019341441239>,0.05
    ,<-0.1088113986073722,-0.0010335688139989977,3.700287237290025>,0.05
    ,<-0.0688550865699358,-0.000987090870483424,3.7303390656350275>,0.05
    ,<-0.033914246843998005,-0.0009060305295045749,3.76609444062429>,0.05
    ,<-0.007883228019114183,-0.0007843495890958649,3.808768233015917>,0.05
    ,<0.005371562317263974,-0.0006379464111907916,3.8569570571494434>,0.05
    ,<0.003306413151408968,-0.0004958648959393332,3.9068886784600565>,0.05
    ,<-0.014327616000923412,-0.0003843751653638069,3.9536496402771046>,0.05
    ,<-0.04528405797153905,-0.0003214665550432936,3.992892074291882>,0.05
    ,<-0.0853441263484594,-0.00031246929412699653,4.022800963382701>,0.05
    ,<-0.12986686411761592,-0.0003490823759506122,4.0455617971255124>,0.05
    ,<-0.17507649183379834,-0.0004137817465162483,4.066932400395111>,0.05
    ,<-0.21711128422691248,-0.00048615066757569695,4.0940071085678635>,0.05
    ,<-0.24999758671314537,-0.000540613609870848,4.131654672005538>,0.05
    ,<-0.2665084263978026,-0.000553250856919802,4.178825576625048>,0.05
    ,<-0.2630844290087231,-0.0005189631477925201,4.228679313362245>,0.05
    ,<-0.24276028697201554,-0.0004541364309354157,4.274332349966333>,0.05
    ,<-0.21212675402714104,-0.0003856199351792067,4.313822718402413>,0.05
    ,<-0.177168538263446,-0.00033059286958739335,4.349552863913796>,0.05
    ,<-0.141759639031601,-0.0002897703431298974,4.384845514466669>,0.05
    ,<-0.10746712211810124,-0.00025671146442657696,4.421229491322955>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
