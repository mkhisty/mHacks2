import React from "react"
import logo from "./images/test.jpg"
import ReactCompareImage from 'react-compare-image';
import { i } from "mathjs";
import c2 from "./images/wamur.png"
import s2 from "./images/wamu2.jpg"
import s3 from "./images/wamu_.jpeg"

export default function OC(){
return <div><div className = "oc">
<p className = "disc">

Pixzip compresses images in an inherently novel way when compared to current compression methods. Currently, most compression algorithms scan the image for patterns, and encode the patterns in a way that takes up much less space than the raw pixels. Instead of doing this through a fixed algorithm, Pixzip's neural network first compresses the image into a tiny latent space. Then, another network is used to decompress it back to original dimensions. Through training, the model learns how to efficiently encode only the valuable data.




</p>
<div className="mcomp">
<p >Original Image vs Pixzip Compressed Image</p>

<div className="compare">
<ReactCompareImage className= "c2" leftImage={s2} rightImage={c2} />;
</div>
</div>
</div>



<div className = "oc">
<div className="mcomp">
<p > Pixzip Compressed Image: 1.2 KB vs JPEG Compressed Image: 1.1KB</p>

<div className="compare">
<ReactCompareImage className= "c2" leftImage={c2} rightImage={s3} />;
</div>
</div>

<p className = "disc">

This method allows images to be compressed to a much greater extent. Even when the JPEG algorithm is set to maximize compression over accuracy, it fails to compress it as much as Pixzip. In addition to that, it still has significantly worse image quality.



</p>









</div>

</div>

}

