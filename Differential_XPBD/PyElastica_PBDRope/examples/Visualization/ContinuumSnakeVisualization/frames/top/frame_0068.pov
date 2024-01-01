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
    ,<-0.1823410303695755,-0.0016758387036561512,1.4332897413449939>,0.05
    ,<-0.14393697294434385,-0.0015523092888597824,1.465294406552037>,0.05
    ,<-0.10394213330396927,-0.0014380872729117533,1.495298218378261>,0.05
    ,<-0.06267876753443663,-0.0013366463399741922,1.5235447999240257>,0.05
    ,<-0.021414794474659403,-0.0012425606476870275,1.5518011044529607>,0.05
    ,<0.01808792658551212,-0.001140467825757005,1.582474970580555>,0.05
    ,<0.05350058170670818,-0.00101059419129613,1.6177879946437506>,0.05
    ,<0.08159689076182539,-0.0008368157649892166,1.6591508336670806>,0.05
    ,<0.09802602275271638,-0.0006174175502894671,1.7063647936545556>,0.05
    ,<0.0981923284465293,-0.00037757619832938836,1.756341851200641>,0.05
    ,<0.08013651894804212,-0.00016110602866080488,1.8029362600013017>,0.05
    ,<0.04681890305889374,-7.128639667518568e-06,1.840183304961847>,0.05
    ,<0.0044833324638902244,6.441059239765774e-05,1.8667572123716254>,0.05
    ,<-0.041191746805356116,5.695127236703311e-05,1.8870859302617913>,0.05
    ,<-0.08629609791449473,-9.81481596017286e-06,1.908657110406208>,0.05
    ,<-0.12670733366516096,-0.00011055844566577713,1.9380863262057153>,0.05
    ,<-0.1571861595190337,-0.00022363983478455388,1.9777008389322084>,0.05
    ,<-0.17396404756071493,-0.0003380101093759033,2.024776667519847>,0.05
    ,<-0.17758547435561006,-0.0004549342082098696,2.0746202803463762>,0.05
    ,<-0.1723496977659059,-0.0005766970211674392,2.1243243301861487>,0.05
    ,<-0.1634826726405034,-0.0006999778911411515,2.1735154738969125>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
