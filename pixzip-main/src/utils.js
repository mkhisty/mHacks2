const MIN_RADIUS = 7.5;
const MAX_RADIUS = 15;
const DEPTH = 2;
const LEFT_COLOR = [0, 232, 255];
const RIGHT_COLOR = [136, 0, 255];
const NUM_POINTS = 750;

  
    // Convert each RGB component to hexadecimal and concatenate
    function getColorFun(r, g, b) {
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
     }
     


  

/**
 * --- Credit ---
 * https://stackoverflow.com/questions/16360533/calculate-color-hex-having-2-colors-and-percent-position
 */
const getGradientStop = (ratio) => {

    return (RIGHT_COLOR[0]-LEFT_COLOR[0])*ratio
};

const calculateColor = (x) => {
    const stop = getGradientStop((x+5)/10);
    return stop;
};

const randomFromInterval = (min, max) => {
    return Math.random() * (max - min) + min;
};

export const pointsInner = Array.from(
    { length: NUM_POINTS },
    (v, k) => k + 1
).map((num) => {

    const x = randomFromInterval(-5, 5);
    const y = randomFromInterval(-5, 5);
    const z = randomFromInterval(-5, 5);

    return {
    idx: num,
    position: [x, y, z],
    color: getColorFun(calculateColor(x),3,252),
    };
});

