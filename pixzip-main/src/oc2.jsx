import React from "react"
import logo from "./images/test.jpg"
import ReactCompareImage from 'react-compare-image';
import s2 from "./images/rock.jpg"
import s1 from "./images/rock.jpg"
import s3 from "./images/rock_.jpeg"

export default function OC2(){
return <div><div className = "oc" style={{height:"125vw"}}>
<p className = "disc2">


Another huge advantage over conventional compression methods is that the performance is dependent on how well training data resembles real life images that need to be compressed. Therefore, the more specific a use case is, the more advantage this method will have. For example, if a platform needs to store thousands of profile pictures, a model trained solely on profile pictures will do exceptionally well.

In this example, a face images dataset had been used for training.




</p>
<div className="mcomp">
<p >Original vs Pixzip Compressed Image</p>

<div className="compare">

<ReactCompareImage className= "c2" leftImage={s2} rightImage={s1} />;

</div>
<p > Pixzip Compressed Image: 200 Bytes vs JPEG Compressed Image: 1.2KB</p>

<div className="compare">

<ReactCompareImage className= "c2" leftImage={s1} rightImage={s3} />;

</div>
</div>

</div>



</div>

}

