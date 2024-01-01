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
    ,<-0.19025841569722968,-0.001890294991076472,5.038313038000545>,0.05
    ,<-0.22600354415082816,-0.0016194406892562636,5.073250325669059>,0.05
    ,<-0.25809310704877503,-0.001336341872609281,5.111573171223939>,0.05
    ,<-0.2823430648182049,-0.001036266723461649,5.1552787550458>,0.05
    ,<-0.29484309842727235,-0.0007323914674810881,5.203670818829634>,0.05
    ,<-0.29316272834196516,-0.00045990784708959916,5.253621760042005>,0.05
    ,<-0.27718053446902047,-0.0002587972568300175,5.300980262887611>,0.05
    ,<-0.24899664003639743,-0.00015262669001637824,5.342267806988028>,0.05
    ,<-0.21214088097137476,-0.00014405862417394178,5.376055006673122>,0.05
    ,<-0.17063151068004914,-0.00021613494899421892,5.403941916636371>,0.05
    ,<-0.12828905611553415,-0.00033825144137755774,5.430550943499127>,0.05
    ,<-0.08935827218353701,-0.0004743047438696912,5.461931638203223>,0.05
    ,<-0.06002730389884565,-0.0005911292831273813,5.502413458853326>,0.05
    ,<-0.04764001786200529,-0.000656227216049685,5.550833209234292>,0.05
    ,<-0.056041513418834364,-0.0006476544622382904,5.600093723328576>,0.05
    ,<-0.08213298528165788,-0.000571664191225588,5.642713319871419>,0.05
    ,<-0.11880665923953182,-0.00045578477095587557,5.676665880470349>,0.05
    ,<-0.15981043114999538,-0.0003273216566774479,5.705252281667417>,0.05
    ,<-0.20119217229620204,-0.00019930636098198107,5.733299562667073>,0.05
    ,<-0.24074159474842072,-7.404190301511505e-05,5.763882694335584>,0.05
    ,<-0.2781524954138479,4.9474910228446457e-05,5.79705219835959>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
