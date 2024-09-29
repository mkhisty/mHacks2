import {BrowserRouter, Routes, Route} from "react-router-dom"
import Home from "./pages/home.js"
import Tech from "./pages/tech.js"
import Api from "./pages/api.js"

export default function App(){
return(
<div>
  <BrowserRouter>
  <Routes>
    <Route index element={<Home/>}/>
    <Route path = "/enterprise" element={<Tech/>}/>
    <Route path = "/api" element={<Api/>}/>
  </Routes>
  </BrowserRouter>
</div>


)

}