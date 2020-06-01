import React, { useState, useRef, useEffect } from "react";
import { Sampler, Players, Sequence, Draw, Transport } from "tone";
import A1 from "./kit/Snare.wav";
import Kick from "./kit/Kick.wav"
import Snare from "./kit/Snare.wav"
import Clap from "./kit/Clap.wav"
import CHH from "./kit/CHH.wav"
import OHH from "./kit/OHH.wav"
import Perc1 from "./kit/Perc1.wav"
import Perc2 from "./kit/Perc2.wav"
import Bass from "./kit/808.wav"

const App = () => {
  const [isLoaded, setLoaded] = useState(false);
  const sampler = useRef(null);

  useEffect(() => {
    let kit = new Players(
      {
        "Kick": Kick,
        "Snare": Snare,
        "Clap" : Clap,
        "CHH": CHH,
        "OHH": OHH,
        "Perc1": Perc1,
        "Perc2": Perc2,
        "Bass": Bass,
      },
      
    ).toMaster();

    let keys = new Players({
			"A" : "./kit/Kick.wav",
			"C#" : ":./kit/Snare.wav",
		}, {
			"volume" : -10,
      "fadeOut" : "64n",
      "onLoad": () => {
        setLoaded(true);
        console.log('Ready');
      }
		}).toMaster();
    /*
   let sampler = new Sampler(
    { A1 },
    {
      onload: () => {
      setLoaded(true);
      }
    }
  ).toMaster();
*/
    let noteNames = ["Kick", "Snare", "Clap", "CHH", "OHH", "Perc1", "Perc2", "Bass"];

		let loop = new Sequence(function(time, col){
			let column = document.querySelector("tone-step-sequencer").currentColumn;
			column.forEach(function(val, i){
				if (val){
					//slightly randomized velocities
					let vel = Math.random() * 0.5 + 0.5;
					kit.get(noteNames[i]).start(time, 0, "32n", 0, vel);
				}
			});
			//set the column on the correct draw frame
			Draw.schedule(function(){
				document.querySelector("tone-step-sequencer").setAttribute("highlight", col);
			}, time);
		}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ,21, 22 ,23 ,24, 25, 26, 27, 28, 29, 30, 31], "16n").start(0);

		//bind the interface
		document.querySelector("tone-transport").bind(Transport);

		Transport.on("stop", () => {
			setTimeout(() => {
				document.querySelector("tone-step-sequencer").setAttribute("highlight", "-1");
			}, 100);
		});


  }, []);

  

  const handleClick = () => sampler.current.triggerAttack("A1");

  return (
    <div>
      <button disabled={!isLoaded} onClick={handleClick}>
        start
      </button>
      <tone-content>
          <tone-transport></tone-transport>
          <tone-step-sequencer></tone-step-sequencer>
        </tone-content>
    </div>
  );
};

export default App;
