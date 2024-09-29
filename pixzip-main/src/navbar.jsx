import React from "react"
import logo from "./images/logo.jpg"
export function Nav(){
return <div className = "nav">

<img className="logo" src = {logo}></img>
<a href = "/api" className="navlink">Get Access</a>
<a href = "/enterprise" className="navlink">Enterprise</a>
<a href = "/" className="navlink">Home</a>

</div>




}

