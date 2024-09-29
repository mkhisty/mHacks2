import React, { useRef,forwardRef} from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere } from "@react-three/drei";
import {pointsInner} from "./utils.js";
import {pow} from "mathjs";

const Landing = () =>{
return <div>
<h1 style = {{color:"#FFFFFF"}} className="title">Shrinking Size, Amplifying Brilliance</h1>

<Canvas className  = "c"  style = {{height:"43vw", backgroundColor:"#121212"}}>
<OrbitControls></OrbitControls>
<directionalLight/>
<pointLight position={[-30,0,-30]} power = {10.0}/>
<Cloud/>
</Canvas>

</div>
}






const randomFromInterval = (min, max) => {
  return Math.random() * (max - min) + min;
};


const Cloud = () => {

  const ref = useRef(null);

  useFrame(({ clock }) => {
    if (ref.current?.rotation) {
      ref.current.rotation.z = clock.getElapsedTime() * 0.5;
      ref.current.rotation.x = clock.getElapsedTime() * 0.5;
      ref.current.rotation.y = clock.getElapsedTime() * 0.5;

    }
  });




console.log(pointsInner)
    return (
<group ref={ref}>
{pointsInner.map(point => <Point  key = {point.idx} position={point.position}  color={point.color} className="1"/>)}
</group>
    );
  };


  const Point = ({position,color})=>{
    const ref = useRef(null);
    let xo = position[0]
    let yo = position[1]
    let zo = position[2]
    let xl = randomFromInterval[1,-1]
    useFrame(({ clock }) => {
      if(clock.getElapsedTime()<17){
      ref.current.position.x = (xo*pow(1.2,-clock.getElapsedTime()))
      ref.current.position.y = yo*pow(1.2,-clock.getElapsedTime())
      ref.current.position.z = zo*pow(1.2,-clock.getElapsedTime())
      }

    });
    return(
  
  <Sphere position={position} args = {[0.1,10,10]} ref={ref}>
      <meshStandardMaterial color = {color} emissive={color} emissiveIntensity={2} roughness={0.5} />
  </Sphere>
  

  
    );
  }

export default Landing